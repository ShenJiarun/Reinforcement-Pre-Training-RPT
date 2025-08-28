import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from rpt import RPTTrainer, RPTModel, RewardSystem, DataProcessor
import warnings
warnings.filterwarnings('ignore')


CONTENT_PROMPT = """You are an AI language model operating in a special, restricted mode. Your ONLY function is to act as a **Literal Next-Token Predictor**.

You do not solve problems, answer questions, or complete thoughts. You only predict the single, most statistically likely token that immediately follows the provided text prefix. This is a purely syntactic and statistical task.

**CRITICAL INSTRUCTION:** DO NOT interpret the meaning of the prefix. DO NOT solve any math problems or answer any questions. Your task is to predict the next token in the sequence, which could be a word, part of a word, a punctuation mark, a space, or a newline character.

---
**EXAMPLE**
**INPUT**
Prefix: The quadratic formula is x = [-b ± sqrt(b^2 - 4ac)] / 2a. To solve the equation 2x^2 + 5x - 3 = 0, we can use this formula. The next step is to

**REASONING (write out your chain-of-thought)**
- The prefix ends with the phrase "The next step is to".
- Grammatically, this phrase is followed by a verb in its base form.
- The context is about using a formula. Common verbs in this context are "identify", "substitute", "plug", "determine".
- "identify" is a very common and logical verb to start the explanation of the process.

**OUTPUT FORMAT**
Reasoning:
The prefix ends with the phrase "The next step is to". Grammatically, this phrase requires a verb in its base form. Given the mathematical context of applying a formula, a common next action is to identify the coefficients from the equation. Therefore, the most probable next token is the word "identify".
Final:
\\boxed{{identify}}
---

**TASK**
Given the following text prefix, perform your function as a Literal Next-Token Predictor.

**INPUT**
Prefix: {CONTEXT}

**REASONING (write out your chain-of-thought)**
- Analyze the very end of the prefix string.
- Consider grammar, syntax, and statistical likelihood of what character or word comes next.
- Explicitly state that you are not solving the problem, but predicting the next token in the text sequence.
- Keep the reasoning concise (no more than 6 lines).

**OUTPUT FORMAT**
Reasoning:
<your reasoning here>
Final:
\\boxed{{<token>}}
"""



def compute_token_entropy(model, tokenizer, text, device='cuda'):
    """
    Compute entropy for each token in the text using the model's predictions.
    
    Returns:
        tokens: List of token strings
        entropies: List of entropy values for each token
        token_ids: List of token IDs
    """
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    entropies = []
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
        
        # Compute entropy for each position
        for i in range(len(tokens) - 1):  # -1 because we predict next token
            # Get probability distribution for next token prediction
            probs = F.softmax(logits[i], dim=-1)
            
            # Compute entropy: -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            probs = probs + epsilon
            entropy = -torch.sum(probs * torch.log(probs)).item()
            entropies.append(entropy)
        
        # Add a placeholder entropy for the last token (no next token to predict)
        entropies.append(0.0)
    
    return tokens, entropies, input_ids[0].tolist()

def analyze_prompts_entropy(model, tokenizer, prompts, device='cuda'):
    """
    Analyze entropy for multiple prompts and return comprehensive statistics.
    """
    all_results = []
    all_entropies = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\nAnalyzing Prompt {idx + 1}...")
        tokens, entropies, token_ids = compute_token_entropy(model, tokenizer, prompt, device)
        
        # Store results
        result = {
            'prompt_idx': idx,
            'tokens': tokens,
            'entropies': entropies,
            'token_ids': token_ids,
            'mean_entropy': np.mean(entropies[:-1]),  # Exclude last token
            'std_entropy': np.std(entropies[:-1]),
            'max_entropy': np.max(entropies[:-1]),
            'min_entropy': np.min(entropies[:-1])
        }
        all_results.append(result)
        all_entropies.extend(entropies[:-1])  # Collect all entropies except last tokens
        
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Mean entropy: {result['mean_entropy']:.4f}")
        print(f"  Std entropy: {result['std_entropy']:.4f}")
        print(f"  Min/Max entropy: {result['min_entropy']:.4f} / {result['max_entropy']:.4f}")
    
    return all_results, all_entropies

def determine_threshold(all_entropies, percentile=75):
    """
    Determine threshold for filtering high entropy tokens.
    
    Args:
        all_entropies: List of all entropy values
        percentile: Percentile to use as threshold (default: 75th percentile)
    
    Returns:
        threshold: Entropy threshold value
    """
    threshold = np.percentile(all_entropies, percentile)
    return threshold

def filter_high_entropy_tokens(results, threshold):
    """
    Filter and identify high entropy (low confidence) tokens based on threshold.
    """
    filtered_results = []
    
    for result in results:
        high_entropy_indices = []
        high_entropy_tokens = []
        
        for i, entropy in enumerate(result['entropies'][:-1]):  # Exclude last token
            if entropy > threshold:
                high_entropy_indices.append(i)
                high_entropy_tokens.append((result['tokens'][i], entropy))
        
        filtered_result = {
            'prompt_idx': result['prompt_idx'],
            'high_entropy_indices': high_entropy_indices,
            'high_entropy_tokens': high_entropy_tokens,
            'total_tokens': len(result['tokens']) - 1,
            'high_entropy_count': len(high_entropy_indices),
            'high_entropy_ratio': len(high_entropy_indices) / (len(result['tokens']) - 1)
        }
        filtered_results.append(filtered_result)
    
    return filtered_results


prompts = [
    """Let $ n(\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\le i ,j\le n)$ satisfying:
(1)For every $1\le i ,j\le n, x_{ij}=max\{x_{i1},x_{i2},...,x_{ij}\} $ or $ x_{ij}=max\{x_{1j},x_{2j},...,x_{ij}\}.$
(2)For every $1\le i \le n$, there are at most $m$ indices $k$ with $x_{ik}=max\{x_{i1},x_{i2},...,x_{ik}\}.$
(3)For every $1\le j \le n$, there are at most $m$ indices $k$ with $x_{kj}=max\{x_{1j},x_{2j},...,x_{kj}\}.$
Let \( n (\geq 2) \) be a positive integer. We aim to find the minimum \( m \) such that there exists \( x_{ij} \) (for \( 1 \leq i, j \leq n \)) satisfying the following conditions:
1. For every \( 1 \leq i, j \leq n \), \( x_{ij} = \max \{ x_{i1}, x_{i2}, \ldots, x_{ij} \} \) or \( x_{ij} = \max \{ x_{1j}, x_{2j}, \ldots, x_{ij} \} \).
2. For every \( 1 \leq i \leq n \), there are at most \( m \) indices \( k \) such that \( x_{ik} = \max \{ x_{i1}, x_{i2}, \ldots, x_{ik} \} \).
3. For every \( 1 \leq j \leq n \), there are at most \( m \) indices \( k \) such that \( x_{kj} = \max \{ x_{1j}, x_{2j}, \ldots, x_{kj} \} \).

To solve this, we need to consider the structure and constraints given by the problem. The solution involves ensuring that the maximum number of indices \( k \) for which \( x_{ik} \) or \( x_{kj} \) is the maximum is minimized.

By analyzing the constraints and constructing examples, it can be shown that the minimum \( m \) satisfying the conditions is:
\[
m = 1 + \left\lceil \\frac{n}{2} \\right\\rceil.
\]

Thus, the minimum value of \( m \) is:
\[
\\boxed{1 + \left\lceil \\frac{n}{2} \\right\\rceil}.
\]""",
    """In an acute scalene triangle $ABC$, points $D,E,F$ lie on sides $BC, CA, AB$, respectively, such that $AD \perp BC, BE \perp CA, CF \perp AB$. Altitudes $AD, BE, CF$ meet at orthocenter $H$. Points $P$ and $Q$ lie on segment $EF$ such that $AP \perp EF$ and $HQ \perp EF$. Lines $DP$ and $QH$ intersect at point $R$. Compute $HQ/HR$.
In an acute scalene triangle \(ABC\), points \(D, E, F\) lie on sides \(BC, CA, AB\), respectively, such that \(AD \perp BC\), \(BE \perp CA\), \(CF \perp AB\). Altitudes \(AD, BE, CF\) meet at orthocenter \(H\). Points \(P\) and \(Q\) lie on segment \(EF\) such that \(AP \perp EF\) and \(HQ \perp EF\). Lines \(DP\) and \(QH\) intersect at point \(R\). We aim to compute \(\\frac{HQ}{HR}\).

Note that \(H\) and \(A\) are the incenter and \(D\)-excenter of \(\\triangle DEF\), respectively. Thus, \(HQ\) is an inradius of \(\triangle DEF\). Let \(R'\) be the reflection of \(Q\) over \(H\). The homothety centered at \(D\) that maps the incircle to the \(D\)-excircle also maps \(R'\) to \(P\), implying that \(D\), \(R'\), and \(P\) are collinear, so \(R' = R\).

Therefore, \(\\frac{HQ}{HR} = 1\).

The answer is \(\\boxed{1}\)."""]

# Load Proxy Model to filter confindent tokens
proxy_tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/date/sjr/Reinforcement-Pre-Training-RPT/model_from_hf/DeepSeek-R1-Distill-Qwen-1.5B")
proxy_model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/date/sjr/Reinforcement-Pre-Training-RPT/model_from_hf/DeepSeek-R1-Distill-Qwen-1.5B")
proxy_model.eval()
proxy_model.to("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Analyze entropy
print("\nComputing token entropy for prompts...")
results, all_entropies = analyze_prompts_entropy(proxy_model, proxy_tokenizer, prompts, device)

# Determine threshold (using 75th percentile as default)
# You can adjust the percentile to be more or less strict
threshold = determine_threshold(all_entropies, percentile=95)

# Filter high entropy tokens
filtered_results = filter_high_entropy_tokens(results, threshold)

results_metrics = {
    'results': results,
    'filtered_results': filtered_results,
    'threshold': threshold,
    'all_entropies': all_entropies
}

separate_prompts = []
labels = []

for index, prompt in enumerate(prompts):
    prompt_tmp = []
    label_tmp = []
    high_entropy_indices = results_metrics['filtered_results'][index]['high_entropy_indices']
    for i in high_entropy_indices:
        if i == 0 or i <= len(prompt) // 4:
            continue
        prompt_tmp.append(CONTENT_PROMPT.format(CONTEXT=prompts[index][:i]))
        label_tmp.append(prompts[index][i])
    labels.extend(label_tmp)
    separate_prompts.extend(prompt_tmp)

# For debugging: print the separate prompts
# print(f"\nSeparate prompts for training: {separate_prompts}")
# print(separate_prompts[0])

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/date/sjr/Reinforcement-Pre-Training-RPT/model_from_hf/DeepSeek-R1-Distill-Qwen-1.5B")
base_model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/date/sjr/Reinforcement-Pre-Training-RPT/model_from_hf/DeepSeek-R1-Distill-Qwen-1.5B")

# Create RPT model with value head
rpt_model = RPTModel(
    base_model=base_model,
    tokenizer=tokenizer,
    add_value_head=True
)

# Setup reward system
reward_system = RewardSystem(
    reward_type="reward",  # Combines accuracy and confidence
    reward_scale=1.0
)

# Prepare your data
data_processor = DataProcessor(tokenizer=tokenizer)

dataset = data_processor.create_dataset(separate_prompts, labels=labels, split_ratio=0.9)
train_dataset, val_dataset = dataset

# Create data loaders
train_loader = data_processor.create_dataloader(
    train_dataset, 
    batch_size=1, 
    shuffle=True
)
val_loader = data_processor.create_dataloader(
    val_dataset, 
    batch_size=1,
    shuffle=False
)

# Setup optimizer
optimizer = torch.optim.AdamW(rpt_model.parameters(), lr=5e-5)

# Create trainer
trainer = RPTTrainer(
    model=rpt_model,
    tokenizer=tokenizer,
    reward_system=reward_system,
    optimizer=optimizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    max_epochs=3,
    output_dir="./rpt_output"
)

# Start training
results = trainer.train()

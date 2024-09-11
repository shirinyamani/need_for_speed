"""You should implement both of these methods.

Vanilla edits should just be a custom generate loop with a huggingface
transformer.

Speculative edits should implement the speculative editing algorithm.

To test these, make sure they work on the prompt provided in the README"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def vanilla_edit(prompt: str, max_tokens: int) -> str:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_tokens,
        num_return_sequences=1,
        attention_mask=attention_mask,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def speculative_edit(prompt: str, max_tokens: int) -> str:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Extract the code block from the prompt
    code_start = prompt.find("```ts") + 6
    code_end = prompt.rfind("```")
    original_code = prompt[code_start:code_end].strip()

    # Tokenize the entire prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Prepare the edit instruction
    edit_instruction = "# This is a comment\n"
    draft_tokens = tokenizer.encode(edit_instruction, add_special_tokens=False)

    generated = input_ids[0].tolist()
    original_code_tokens = tokenizer.encode(original_code, add_special_tokens=False)

    with torch.no_grad():
        # Use the entire original code as the initial speculation
        speculation = original_code_tokens + draft_tokens + original_code_tokens
        
        prefix = torch.tensor([generated])
        outputs = model(prefix)
        
        for i, token in enumerate(speculation):
            next_token_logits = outputs.logits[0, -1, :]
            predicted_token = torch.argmax(next_token_logits).item()

            if predicted_token != token:
                # Model disagrees with speculation, start generating from here
                generated.extend(speculation[:i])
                break
            
            # Move forward one step
            prefix = torch.cat([prefix, torch.tensor([[token]])], dim=1)
            outputs = model(prefix[:, -1:], past_key_values=outputs.past_key_values)
        else:
            # If the loop completes without breaking, all tokens were accepted
            generated.extend(speculation)

        # Continue generating tokens up to max_tokens
        while len(generated) - len(input_ids[0]) < max_tokens:
            next_token_logits = outputs.logits[0, -1, :]
            predicted_token = torch.argmax(next_token_logits).item()
            generated.append(predicted_token)
            
            prefix = torch.tensor([[predicted_token]])
            outputs = model(prefix, past_key_values=outputs.past_key_values)

    result = tokenizer.decode(generated, skip_special_tokens=True)

    # Reconstruct the prompt structure
    edited_code = result[len(prompt):]
    final_result = prompt[:code_start] + edited_code + prompt[code_end:]

    return final_result


test_prompt = '''````txt
Please add a single comment
```ts
export default function Visualization() {
  const [instanceIdInputs, setInstanceIdInputs] = createSignal<
    InstanceId[] | null
  >(null);
  const [storedInput, setStoredInput] = createSignal<string>("");
  const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
    []
  );
  const [shouldRefreshGold, setShouldRefreshGold] =
    createSignal<boolean>(false);
  const [showGold, setShowGold] = createSignal<boolean>(false);
  const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
    string | undefined
  >(undefined);
  const [goldInstances, setGoldInstances] = createSignal<
    {
      sessionId: string;
      email: string | undefined;
      requestId: string | undefined;
      dateAdded: Date;
      type: $Enums.CppGoldExampleType;
    }[]
  >([]);
}
```
```ts
````
'''
import time

def time_function(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

vanilla_result, vanilla_time = time_function(vanilla_edit, test_prompt, 50)
speculative_result, speculative_time = time_function(speculative_edit, test_prompt, 50)

print("Vanilla Edit Result:")
print(vanilla_result)
print(f"Vanilla time: {vanilla_time:.4f} seconds\n")

print("Speculative Edit Result:")
print(speculative_result)
print(f"New time: {speculative_time:.4f} seconds")

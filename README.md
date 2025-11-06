# The 100-Line AI agent that solves GitHub issues with text-edit tool
> A ~100-line SWE scaffold, now with a string-replacement editor—fewer rounds, and performance close to mainstream frameworks on SWE-bench. Simpler systems generalize better and transfer more easily to other tasks—give it a try!
## Introduction
The **[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)** which is 100x smaller than [SWE-agent](https://github.com/swe-agent/swe-agent) achieves an impressive **~65%** solve rate, proving that a **lean software scaffold can still be strong**.

However, case studies reveal clear weaknesses in using only `bash` for code editing—`sed`’s line-based design makes multi-line changes cumbersome and brittle, complex cross-block edits are fragile under regex/escaping, and the model may even try to improvise its own edit tools. 

In mini-swe-agent-plus, we add a minimal string-replacement editor, which performs text replacements if only one is matched. Under the same evaluation setup, mini-swe-agent-plus **significantly reduces average rounds** and narrows the gap with SWE-agent on models like Claude 4 Sonnet.

| Model | Software scaffold | SWE-bench Verified | Avg. Rounds | 
|------------------|-----------------------|--------------|-------------| 
|Claude 4.5 Sonnet | mini-swe-agent      | 70.6\% | — |
|**Claude 4.5 Sonnet** | **mini-swe-agent-plus** | **71.8\%** | — |
| Claude 4 Sonnet| Claude Private | 72.7 | — | 
| Claude  4 Sonnet | OpenHands | 70.4 | — | 
| Claude  4 Sonnet | SWE-agent | 66.6 | — | 
| Claude  4 Sonnet| mini-swe-agent | 64.93% | 79.1 | 
| **Claude  4 Sonnet** | **mini-swe-agent-plus** | **67.0%** | **67.3** |
| Claude  3.7 Sonnet | SWE-agent | 58.2% | — |
| Claude  3.7 Sonnet| mini-swe-agent | 52.8%  | 62.9 |
| **Claude  3.7 Sonnet** | **mini-swe-agent-plus** | **54.6%** | **48.7** |

## Scalable training performance
We collected 66k trajectories based on [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) and study the model performance with different data scaling. 
The figure reports solve rate on SWE-bench Verified for the Qwen3-8B model. Empirically, performance increases approximately linearly with the logarithm of the data scale.
Klear-Agent-8B significantly outperforms other ~8B models and even matches the performance of some open 32B systems. 

<p align="left">
  <img src="https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k/resolve/main/swe_bench_scaling_grid.svg" width="66%" alt="image" />
</p>

| Method/Model            | Params | Agent Framework | SWE-bench Verified (%) |
|-------------------------|:------:|-----------------|:----------------:|
| SWE-agent-LM-7B         | 7B     | SWE-agent       | 15.2             |
| SWE-Mirror-LM-7B        | 7B     | OpenHands       | 22.8             |
| SWE-gym-32B             | 32B    | OpenHands       | 20.6             |
| Skywork-SWE-32B         | 32B    | OpenHands       | 38.0             |
| DeepSWE-32B-Preview     | 32B    | OpenHands       | 42.2             |
| SWE-Mirror-LM-32B       | 32B    | OpenHands       | 52.2             |
| SWE-fixer-72B           | 72B    | SWE-Fixer       | 32.8             |
| Lingma-SWE-GPT-72B      | 72B    | SWE-Syninfer    | 32.8             |
| **Klear-Agent-8B-SFT**   | 8B     | **mini-swe-agent-plus**   | 39.0             |

The collected trajectories are made openly available at [🤗 SWE-smith-mini_swe_agent_plus-trajectories-66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k).
## Usage
https://github.com/Kwai-Klear/mini-swe-agent-plus
#### Install
```
git clone https://github.com/Kwai-Klear/mini-swe-agent-plus
cd mini-swe-agent-plus
pip install -e .
```

#### Evaluation on SWE-bench Verified
`src/minisweagent/run/extra/swebench_pool_way.py` replicates `src/minisweagent/run/extra/swebench.py` using multiprocessing, offering somewhat improved stability over the original multi-threaded implementation. The usage is identical, please refer to [mini-swe-agent's doc](https://mini-swe-agent.com/latest/usage/swebench/) for more details.
```

# Using llm model service
config=src/minisweagent/config/extra/swebench_add_edit_tool.yaml
python src/minisweagent/run/extra/swebench_pool_way.py \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --subset verified \
    --split test \
    --output=eval_results/the_outpath \
    --config $config
    --workers 4


# Using multiple vllm server:
export local_vllm_server_ips_filename=_the_path_to_vllm_server_ips_.txt
#`_the_path_to_vllm_server_ips_.txt` each node ip per line e.g. `http://127.0.0.1:8000/v1`

config=src/minisweagent/config/extra/swebench_add_edit_tool.yaml
python src/minisweagent/run/extra/swebench_pool_way.py \
    --model hosted_vllm/hosted_model_name \
    --subset verified \
    --split test \
    --output=eval_results/the_outpath \
    --config $config
    --workers 4
```
More models could be easily extented via the `litellm`, please refer to `src/minisweagent/models/litellm_model.py` for more details.
It is much better pre-featch the docker images before the evaluate on SWE-bench verified. Raise an issue when error reported. 


## Acknowledgements
mini-swe-agent-plus is built on top of mini-swe-agent. The training dataset is constructed from SWE-smith, and we gratefully acknowledge the SWE-bench team for their benchmark and infrastructure. Our model is trained with Qwen3-8B via ms-swift.



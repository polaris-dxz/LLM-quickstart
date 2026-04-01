# LLM-quickstart 学习心得

下面是我跟着 [DjangoPeng/LLM-quickstart](https://github.com/DjangoPeng/LLM-quickstart) 动手跑代码时的一些**体会**，方便以后复习，也和课程作业对照着看。

---

## 我对这个仓库的整体感受

- **它在做什么**：把「大模型怎么用、怎么训、怎么省显存」拆成一个个 Notebook，从 `transformers` 最基础的 Pipeline，到微调、PEFT、量化、DeepSpeed，是一条**循序渐进**的线，而不是散乱的脚本堆砌。
- **它适合怎么学**：我更多是 **边跑边改参数**、看 loss / 显存变化，比纯看理论印象深；但**真正吃透**仍然要有一台 **GPU 环境**把关键几本跑完，否则容易停在「看懂一半」。
- **硬件上的清醒认识**：很多笔记在 **Mac 上只能跑前半段**；一碰到 **bitsandbytes、AutoGPTQ、QLoRA 全链路、DeepSpeed**，就回到 **Linux + NVIDIA** 才现实。README 里把 **通用 `requirements.txt`** 和 **`requirements-linux-gpu.txt`** 分开，我后来才理解是为了避免在本地硬装装不上的轮子。

---

## 环境与习惯方面

- **Python 3.10 + 依赖锁定**：跟着 `requirements.txt` 装，能减少「我这边能跑你那边报错」的差；有精力的话用 **uv** 装会快不少。
- **Jupyter**：我习惯用 Lab 分段跑、改参数；Notebook 里经常混着「数据探索 → 训练 → 画图」，适合**复盘**但不适合直接当生产代码，所以课程里让把 `gen_dataset` 改成 `.py` 我理解为：**把可复用逻辑抽成脚本**。
- **`.env` 与 API**：做数据构造、调用 OpenAI 兼容接口时，**密钥和 base URL 放环境变量**，不要写进仓库；换第三方中转时，还要注意 **httpx / openai / langchain** 版本是否匹配，否则会出现一些「看起来是配置错、其实是依赖冲突」的坑。

---

## `transformers/`：从用到训

- **Pipeline**：我体会到的是：同一个任务，**预处理 + 模型 + 后处理**怎么接在一起；进阶那本则让我知道 Pipeline 还能**批处理、扩展**，不是只能 `pipeline("...")` 一行了事。
- **微调**：自己准备 `datasets`、写 `Trainer` 参数后，才理解「数据长什么样」比「模型名多酷」更决定能不能训起来；**抽取式 QA** 那本让我对 **span 标注、指标** 有了具体画面，而不是停留在概念上。

**心得**：能独立跑通一个小微调，**比**记住很多类名更有用；遇到报错时，先看 **数据字段名、tokenizer、max_length** 往往比先怀疑模型坏了更有效。

---

## `peft/`：LoRA 与 QLoRA

- **LoRA（Whisper 那类）**：我理解了「只训少量 adapter 参数」在工程上意味着什么——**省显存、好合并、好分发**。
- **QLoRA + ChatGLM3**：`peft_qlora_chatglm.ipynb` 让我把 **4bit 量化、LoRA、Trainer** 串起来；**广告 / adgen** 那条线则让我看到「领域数据从 HF 拉下来 → tokenize → 训练」的完整流程。
- **和全精度 LoRA 的取舍**：QLoRA 是为了在**单卡**上把大模型训起来；若作业要求 **纯 LoRA**，需要看清是 **fp16 全模型 + LoRA** 还是 **4bit + LoRA**，两者依赖和显存不是一个量级。

**心得**：`PeftModel.from_pretrained` 加载 adapter 的路径，**一定要和训练保存路径一致**；推理时基座 + adapter 的配对，是我后面改 `inference` / Gradio 时最核心的一环。

---

## `quantization/`：量化在解决什么问题

- **bitsandbytes**：8bit/4bit 加载、和训练结合，我意识到是在**换显存**和**换精度**之间做权衡。
- **GPTQ / AWQ**：更偏「**训好后再压权重**」的推理侧思路，和 QLoRA「**训练期就 4bit**」不是同一件事。

**心得**：这些实验在 **没有 CUDA 的 Mac** 上经常**直接跑不了**；不是「我配错了」，而是**工具链本身就不支持或极难支持**。接受这一点，就**把重实验放到云 GPU**，本地只做文档和轻量调试，心态会稳很多。

---

## `chatglm/`：从造数据到上线一条龙

- **造数据**：`gen_dataset.ipynb` 让我体会到 API + LangChain 如何**批量生成监督数据**，以及 CSV 落盘、时间戳管理这种**工程细节**。
- **领域微调**：`qlora_chatglm3.ipynb` 把 `data/` 下 CSV 和 ChatGLM3 接起来；**没有 GPU 时**，第一格检查 CUDA 会报错，我学会了**先看 `torch.cuda.is_available()` 再决定能不能往下跑**。
- **推理对比**：`chatglm_inference.ipynb` 让我直观看到**同一问题下基座 vs 微调后**的差异。
- **Gradio**：`chatbot_webui.py` 虽小，但说明「**模型加载逻辑**」和「**界面**」可以拆成：先保证推理函数对，再套一层 UI。

**心得**：这条线最接近「**业务数据 → 微调 → 演示**」；若课程作业要求接到 **basic_demo / WebUI**，本质是把我在这学到的 **`PeftModel` 加载方式**复制到 Gradio 的 `model` 或 `fn` 里。

---

## `deepspeed/`：大模型训练的另一套工具

- **ZeRO-2 / ZeRO-3**：`ds_config_*.json` 让我看到 **optimizer / 参数分片、offload** 是怎么写进配置的；`train_on_one_gpu.sh` 则把 **deepspeed 命令行 + 翻译任务** 串成一条可执行命令。
- **幻灯片里的 OOM**：CPU 侧报错时，**GPU 还很空**，说明问题在 **ZeRO 初始化或 offload 策略**，不是「我显卡不够」这么简单。

**心得**：DeepSpeed 不是「装完就能训 11B」；**改配置、看 CPU 内存、必要时 NVMe offload**，是单独一套功课，和「会写 PyTorch 训练循环」不完全重叠。

---

## `llama/`、`langchain/`（可选）

- **Llama2**：指令微调那条线，和 ChatGLM 专题**互补**，都是「开源模型 + 指令数据」。
- **LangChain**：我更多把它当成「**应用层**怎么把模型、链、工具串起来」，和「**训练**」是两条线，但后面做 RAG、Agent 会再碰到。

---

## 一些跨笔记本的通用收获

- **会查 Hugging Face 文档**：`AutoModel`、`TrainingArguments`、PEFT、量化参数，**对照官方说明**比死记笔记可靠。
- **显存不是「一个模型名」决定的**：batch、长度、精度、是否量化、是否 LoRA，**叠在一起**才是显存。
- **环境分离**：开发用 Mac，训练用 Linux 云，**同一仓库 + 同一 `requirements`**，减少「我本地能跑、服务器不能跑」的差异。

---

## 一句话总结

这个仓库帮我把「**从会调 Pipeline，到会微调、会 PEFT、会碰量化、知道 DeepSpeed 是干什么的**」串成一条线；**心得**上最大的收获是：**分清哪些步骤必须在 GPU Linux 上完成**，以及**微调产物（adapter）如何被推理和界面加载**，这是后面接作业、接项目时反复要用到的骨架。

---

*若你也在用本仓库，可以把上面的「我」改成你自己的经历，或把某本 Notebook 的踩坑补成一小节，这份文档会更有用。*

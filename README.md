# FET (**F**ine-Grained **E**ntity **T**yping)

该 **Repo** 主要用于 LLM 在实体分类任务的性能研究。

> 如果您需要直接使用 LLM，或者微调后进行各种 benchmark 的评估，这个仓库可能会带来帮助

<br/>
<br/>

> [!IMPORTANT]
> 整个流程分为 3 步：
> - 准备指令微调数据集（可选）
> - 在数据集上进行下游任务（FET）微调（可选）
> - 在 benchmark 上评估性能

<br/>
<br/>

## Stage1

为了实现数据集的统一性，我的方案全部采用 HuggingFace Repo [DGME/figer](https://huggingface.co/datasets/DGME/figer) 的格式，即：

![](/assets/1.png)


对于 **每条数据**，满足以下规则

```yaml
features:
    - name: mention_span
      dtype: string
    - name: left_context_token
      sequence: string
    - name: right_context_token
      sequence: string
    - name: y_str
      sequence: string
```


<br/>
<br/>

因此第一步，**你需要自己将原始数据集整理成上述形式** (只需要完成这一步，后续都能自动化处理)，推荐处理好：

* 使用 `push_to_hub` 函数将您的数据集上传到 HuggingFace
* 然后将所有实体的类别的文件，放置在 `modules/datasets_classes`，可以参考 [figer.txt](modules/datasets_classes/figer.txt)
* 在 `modules/cls.py` 文件中，加入数据集名与所有类别文件的位置的映射，具体参考字典 `NAME_CLS_MAPPING`

<br/>
<br/>

## Stage2

#### LLaMA Factory 数据集构建
接下来，参考 `/scripts/prepare_sft.bat`：

```bat
@echo off

set DATA_PATH = "DGME/figer"
set CLS_PATH = "modules/datasets_classes/figer.txt"
set SPLIT = "test"
set ORDER = 0


python prepare_sft.py ^
    --data_path %DATA_PATH% ^
    --cls_path %CLS_PATH% ^
    --split %SPLIT% ^
    --order %ORDER%
```

这里介绍几个重要的参数：
<!-- > - **MODEL_ID**：用于指定 LLM 信息，既可以是 HuggingFace 远程仓库名，如：`Qwen/QwQ-32B-Preview` ；也可以是您本地 HuggingFace 模型的位置，如：`E:/pretrained_models/Qwen/Qwen2___5-7B-Instruct-GPTQ-Int4`
>
> <br/> -->
> [!WARNING]
> 
> - **DATASET_PATH**：stage1 过后您的数据集
> 
>   - 可以是 HuggingFace 远程仓库名：如 **DGME/figer**
>   - 或者，本地**具有 satge1 上述特征**的 HuggingFace 数据集：如 **E:/pretrained_models/datasets/figer/figer/1.0.0**
> - **CLS_PATH**：该数据集所有类别的标签所在的地址
>
>   - 如 [figer.txt](modules/datasets_classes/figer.txt)，期望是一个 **python list** 的形式，使用 `print(str(xx_list))` 即可获得
>
> - **split**：指定需要该数据集的哪部分来构造指令数据集(意同 [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 的 split 参数)
> - **order**：指定问题模板中，给定的类别出现的顺序，有以下几种选择：
>
>   - **0**：父标签在前，子标签在后，适用于 "/person/artist, /person" 这种标签，否则会触发报错（[TODO]之后考虑更温和的处理方式）
>
>       - 结果为：**/person /person/artist** 
>   - **1**：子标签在前，父标签在后，
>
>       - 结果为：**/person/artist /person**
>
>   - **2**：正向字典序，例如对于 "/artist, /building"
>
>       - 结果为：**/artist /building** 
>   - **3**：反向字典序，例如对于 "/artist, /building"
>
>       - 结果为：**/building /artist** 
>
>   - **4**：随机顺序
> <br/>
> <br/>
> 最终的呈现结果可以参考：
> ```json
>{
>   "instruction":
>       "[Task]: Fine-grained entity classification\n[sentence]: A handful of professors in the UW Department of Chemistry are being recognized by the American Association for the Advancement of Science ( AAAS ) for their efforts and contributions to the scientific community .\n[entity]: UW\n[entity types]: \/ person musician artist director author athlete actor architect doctor politician soldier engineer monarch coach religious_leader terrorist location city country cemetery province body_of_water county bridge language organization company sports_league sports_team educational_institution airline terrorist_organization fraternity_sorority people ethnicity written_work software product computer weapon airplane ship spacecraft car instrument mobile_phone engine_device camera government government political_party park government_agency art film broadcast_program game geography island mountain glacier music military train rail railway building hospital airport sports_facility restaurant hotel theater power_station library dam event military_conflict attack natural_disaster terrorist_attack sports_event election protest title award law astral_body internet website disease chemistry news_agency time transportation road education educational_degree department transit broadcast_network broadcast tv_channel religion religion finance currency stock_exchange food livingthing animal living_thing god metropolitan_transit transit_line play body_part medicine medical_treatment drug symptom newspaper computer algorithm programming_language visual_art color biology\n[Classification Result]: \n[Warning]: Just output nothing except entity types above, separate them by one space, there may be more than one answer",
>   "output":"organization educational_institution",
>   "input":""
> }
> ```

<br/>
<br/>
<br/>
<br/>

之后，运行 `/scripts/sft.bat`，一些重要参数如下：

| 参数 | 解释|
|---|---|
|MODEL_ID | 模型 HuggingFace 的远程仓库名，或者本地仓库名|
|DATASET_ID | 需要在 LLaMA Factory 中的 `dataset_info.json` 中指定的数据集名字，可以参考 [Doc](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md) 以及当前配置 [dataset_info.json](LLaMA-Factory/data/dataset_info.json) 的 `figer_test`
|OUTPUT_DIR| SFT 模型的输出位置|
|TEMPLATE| 模型采用的对话模板 |
|FINETUNING | 微调方法，如 "lora", "qlora" 等|

> 推荐在 `/scripts/sft.bat` 调整更详细的参数

<br/>
<br/>

## Stage3 

模型评估阶段，评估指标使用的是 `strict Acc`，以及 `macro f1`，`micro f1`，运行 `/scripts/evaluate.bat` 即可。
# LLM Choice Workflow

这是一个最小可运行的 Part A + Part B 闭环：

- Part A：输入商品列表文本，由 LLM 选 1 个商品
- Part B：输入货架图片，由 LLM 选 1 个商品
- 两部分统一输出到同一个 JSON 文件

## 文件

- `run_choice_workflow.py`：主流程，调用 OpenRouter
- `make_demo_shelf.py`：生成 demo 货架图
- `demo_products.json`：demo 商品列表
- `.env.openrouter`：远端保存的 API key 文件，不纳入版本控制

## 远端运行示例

```bash
cd /root/autodl-tmp/llm_workflow
/root/miniconda3/bin/python make_demo_shelf.py
/root/miniconda3/bin/python run_choice_workflow.py --mode both
```

默认输出文件：

```bash
/root/autodl-tmp/llm_workflow/demo_run_output.json
```

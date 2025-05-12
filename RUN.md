# Usage

```bash
# python version > 3.12
pip install -r requirements.txt
# 前端运行在 http://127.0.0.1:7860
python demo.py
```

- 前置准备：VPN、Android模拟器
- 使用的模型在`config.py`中修改，目前为`gpt-4o-mini`
- OmniParser和ImageEmbedding服务可用API如下
```bash
# ImageEmbedding Feature_URI(config.py)
GET /available_models - Get list of available models
POST /set_model - Set the model to use
POST /extract_single/ - Extract features from a single image
POST /extract_batch/ - Batch extract features from multiple images
GET /model_info - Get current model information
GET /benchmark/ - Run performance tests
# OmniParser Omni_URI(config.py)
POST /process_image/ - Process image and return parsing results
```
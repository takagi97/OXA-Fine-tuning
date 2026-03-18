cd 2-fine-tuning/LLaMA-Factory
pip install -e ./
pip install -e ".[torch,metrics]"
cd 2-fine-tuning/transformers-4.51.0
pip install -e ./
pip install deepspeed==0.16.9
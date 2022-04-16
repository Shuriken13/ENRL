# shell command for reference

# ENRL on Adult dataset
'python main.py --model_name ENRL --dataset Adult --rule_len 5 --rule_n 40 --es_patience 200 --op_loss 1e-05'

# ENRL on Credit dataset
'python main.py --model_name ENRL --dataset Credit --rule_len 5 --rule_n 40 --es_patience 200 --op_loss 1'

# ENRL on RSC2017 dataset
'python main.py --model_name ENRL --dataset RSC2017 --rule_len 3 --rule_n 40 --es_patience 50 --op_loss 1e-05'

# ENRL on Synthetic dataset
'python main.py --model_name ENRL --dataset Synthetic --rule_len 5 --rule_n 40 --es_patience 200 --op_loss 1'

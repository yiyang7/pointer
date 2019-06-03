# # mymodel combine_all 30 epoch
# python run_summarization.py --mode=train --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=mymodel_combine_all_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.15 --fine_tune=False --train_size=280380 --use_doc_vec=True --lr=0.15
# # convert_to_coverage mymodel combine_all
# python run_summarization.py --mode=train --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=mymodel_combine_all_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.15 --fine_tune=False --train_size=280380 --use_doc_vec=True --lr=0.15 --coverage=1 --convert_to_coverage_model=1
# # decode mymodel combine_all
# python run_summarization.py --mode=decode --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=mymodel_combine_all_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.15 --max_enc_steps=400 --max_dec_steps=120 --use_doc_vec=True --coverage=1 --single_pass=1
# fine_tune pointer combine_all  3 epoch
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.03
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.05
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.3
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.5
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.03
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.05
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.3
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.5
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.03
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.05
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.3
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.5
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.03
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.05
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.3
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.5

# convert to coverage fine_tune pointer combine_all
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.03 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.05 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.1 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.3 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=20 --lr=0.5 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.03 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.05 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.1 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.3 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=100 --min_dec_steps=10 --lr=0.5 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.03 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.05 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.1 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.3 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=20 --lr=0.5 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.03 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.03 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.05 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.05 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.1 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.1 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.3 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.3 --coverage=1 --convert_to_coverage_model=1
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.5 --fine_tune=False --train_size=28017 --use_doc_vec=False --max_dec_steps=50 --min_dec_steps=10 --lr=0.5 --coverage=1 --convert_to_coverage_model=1


# decode fine_tune pointer combine_all
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.03 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.05 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.1 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.3 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_20_lr_0.5 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.03 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.05 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.1 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.3 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_100_min_dec_steps_10_lr_0.5 --max_enc_steps=300 --max_dec_steps=100 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.03 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.05 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.1 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.3 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_20_lr_0.5 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.03 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.05 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.1 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.3 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test_* --vocab_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/vocab --log_root=log --exp_name=fine_tune_exp_hidden_64_emb_32_batch_16_max_enc_steps_300_max_dec_steps_50_min_dec_steps_10_lr_0.5 --max_enc_steps=300 --max_dec_steps=50 --coverage=1 --single_pass=1



# hidden_dim = [32, 64, 128]
# emb_dim = [16, 32, 64 ]
# batch_size = 16
# max_enc_steps= 300
# max_dec_steps= [50, 100]
# beam_size=4
# min_dec_steps= [10, 20]
# vocab_size = 50000

# lr = [0.001, 0.01, 0.1, 1 ]
# adagrad_init_acc = 0.1
# rand_unif_init_mag = 0.02
# trunc_norm_init_std = 1e-4
# max_grad_norm = 2.0
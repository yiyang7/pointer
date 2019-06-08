# repeat the (pre-trained initialized) full training vs fine-tuning on a small dataset, e.g. 1000 relationships (to see whether fine tuning is helpful on small data set)

# Evaluation: 
# lead-3

# # pre-trained
# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_relationships_exp --fine_tune=False --train_size=9520 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_relationships_exp --train_size=9520 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# # pre-trained + full tuning
# # train
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_relationships_exp --fine_tune=False --train_size=9520
# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_relationships_exp --fine_tune=False --train_size=9520 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_relationships_exp --train_size=9520 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# # pretrained + fine tuning
# # train
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_fine_tune_relationships_exp --fine_tune=True --train_size=9520
# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_fine_tune_relationships_exp --fine_tune=True --train_size=9520 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_fine_tune_relationships_exp --train_size=9520 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# Exp 1
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_doc_vec_combine_exp --use_doc_vec=True --train_size=93460
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_doc_vec_combine_exp --use_doc_vec=True --train_size=93460 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k_mymodel/processed_combine_all/finished_combine_all_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_doc_vec_combine_exp --use_doc_vec=True --train_size=93460 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1
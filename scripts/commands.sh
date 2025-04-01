# run only training
python src/train.py task_name=task_name tags="[first_tag, second_tag]"

# run only testing
python src/eval.py task_name=eval data.redo_preprocess=False ckpt_path=ckpt_path tags="[first_tag, second_tag]"

# run training and testing
python src/train.py task_name=task_name test=True tags="[first_tag, second_tag]"
python src/train.py task_name=augmentation test=True tags="[train+test, baseline]" data.processed_dir=data/domain/processed_knn_baseline data.redo_preprocess=False data.batch_size=1 data.augmentation_mode=baseline model.augmentation_mode=baseline model.processed_dir=data/domain/processed_knn_baseline
python src/train.py task_name=augmentation test=True tags="[train+test, pseudo_batch]" data.processed_dir=data/domain/processed_knn_pseudo_batch data.redo_preprocess=True data.batch_size=1 data.augmentation_mode=pseudo_batch_effect model.augmentation=baseline model.processed_dir=data/domain/processed_knn_pseudo_batch
python src/train.py task_name=augmentation test=True tags="[train+test, importance]" data.processed_dir=data/domain/processed_knn_baseline data.redo_preprocess=False data.batch_size=1 data.augmentation_mode=baseline model.augmentation_mode=advanced model.processed_dir=data/domain/processed_knn_baseline

# run hparam search
python src/train.py -m task_name=hparam_search test=True hparams_search=optuna data.batch_size=1 data.redo_preprocess=True tags="[first_tag, second_tag]"
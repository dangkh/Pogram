# baseline
python train.py
python test.py --val_all

# using entity
python train.py --use_entity
python test.py --val_all --use_entity

# using enriched entity
python train.py --use_entity --use_EnrichE
python test.py --val_all --use_entity --use_EnrichE

# using entity and alter title type 0 
python train.py --use_entity --genAbs
python test.py --val_all --use_entity --genAbs

# using entity and alter title type 0 
python train.py --use_entity --use_EnrichE --genAbs
python test.py --val_all --use_entity --use_EnrichE --genAbs

# using entity and alter title type 1
python train.py --use_entity --use_EnrichE --genAbs --genType 1
python test.py --val_all --use_entity --use_EnrichE --genAbs --genType 1


# meta-learning-with-BERT

## 0.Install LEAP
install leap package from https://github.com/amzn/metalearn-leap.git

## 1.Download XQuAD dataset 
> git clone https://github.com/deepmind/xquad

## 2.Train on XQuAD with meta-learning
> XQuAD training dir should contain all languages used to meta-learn

    bash run_meta.sh <XQuAD training dir>
## 3.Test on XQuAD  
> testing lang =en, de, es, ...

    bash eval_xquad.sh <XQuAD testing dir> <model dir> <testing lang>

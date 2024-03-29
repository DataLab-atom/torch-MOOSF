# LSF --- pytorch

### evn 

```
conda create -n MOOSF 3.8.11   
source activate MOOSF 
pip install -r requirements.txt
```

### LSF train and test
```
python main.py --lr 10 --ce --bs --MOOSF
python main.py --lr 50 --ce --bs --MOOSF
python main.py --lr 100 --ce --bs --MOOSF

python main.py --lr 10 --bs --kps --MOOSF
python main.py --lr 50 --bs --kps --MOOSF
python main.py --lr 100 --bs --kps --MOOSF

python main.py --lr 100 --ce_drw --ldam_drw --MOOSF
python main.py --kps --shike --MOOSF
```


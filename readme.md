# LSF --- pytorch

### evn 

```
conda create -n MOOSF 3.8.11   
source activate MOOSF 
pip install -r requirements.txt
```

### LSF train and test
```
python main.py --ce --bs --MOOSF
python main.py --bs --kps --MOOSF

python main.py --ce_drw --ldam_drw --MOOSF
python main.py --kps --shike --MOOSF
```


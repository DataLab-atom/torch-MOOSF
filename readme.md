# LSF --- pytorch

### evn 

```

conda create -n LSF 3.8.11   
source activate LSF 
pip install -r requirements.txt

```

### LSF train and test
```
python main.py --ce --bs --LSF
python main.py --bs --kps --LSF
python main.py --ce_drw --ldam_drw --LSF
python main.py --kps --shike --LSF
```


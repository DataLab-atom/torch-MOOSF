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
python main.py --ce_drw --kps --LSF
python main.py --bs --kps --LSF
```


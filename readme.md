# LSF --- pytorch

### evn 

```
conda create -n LSF 3.8.11   
source activate LSF 
pip install -r requirements.txt
```

### LSF train and test
```
python main.py --ir 10 --ce --bs --LSF
python main.py --ir 50 --ce --bs --LSF
python main.py --ir 100 --ce --bs --LSF

python main.py --ir 10 --bs --kps --LSF
python main.py --ir 50 --bs --kps --LSF
python main.py --ir 100 --bs --kps --LSF

python main.py --ir 100 --ce_drw --ldam_drw --LSF
python main.py --kps --shike --LSF
```


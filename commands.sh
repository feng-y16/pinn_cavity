# ground truth
python main.py -n pinn -l l2 -i 10000 -ntr 20000 -gi 100
# ag1
python main.py -n pinn -l ag -i 10000 -ntr 200 -gi 2000
# pinn1
python main.py -n pinn -l l2 -i 10000 -ntr 3200 -gi 100
# ag2
python main.py -n pinn -l lp -i 10000 -ntr 400 -gi 2000
# pinn2
python main.py -n gfnn -l l2 -i 10000 -ntr 400 -gi 100
# just for small testing
python main.py -n pinn -l l2 -i 10 -ntr 10 -gi 100
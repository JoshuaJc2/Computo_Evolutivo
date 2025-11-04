*Para pasarle el ejemplar especifico a un algoritmo especifico*
python3 src/ils.py --ejemplar src/Ejemplares/Easy1.txt --seed 42 --max_iter 1000 --max_iter_local 1000

python3 src/ils.py --ejemplar src/Ejemplares/SD2.txt --max_iter 1000 --max_iter_local 1000

python3 src/memetico.py --ejemplar src/Ejemplares/SD2.txt --pop_size 100 --max_iter 100 --local_apply_prob 0.2 --local_max_iter 100


python3 src/SA.py --ejemplar src/Ejemplares/SD2.txt --max_iter 1000

python3 src/memetico.py --ejemplar src/Ejemplares/Easy1.txt --max_iter 1000 --max_iter_local 1000


----------------------------------------

python src/experimentacion.py --num_ejecuciones 3


python src/experimentacion.py --ejemplar src/Ejemplares/Easy1.txt --num_ejecuciones 3

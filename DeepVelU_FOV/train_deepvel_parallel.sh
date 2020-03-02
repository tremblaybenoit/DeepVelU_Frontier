python train_deepvel_unet_k2_parallel.py -a start -e 10 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e10
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 15 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e25
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 5 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e30
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 5 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e35
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 5 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e40
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 5 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e45
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python train_deepvel_unet_k2_parallel.py -a continue -e 5 -o network/deepvel -g 2 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
cp -r network network_e50
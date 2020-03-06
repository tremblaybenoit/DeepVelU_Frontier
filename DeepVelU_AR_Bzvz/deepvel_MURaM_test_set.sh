python deepvel_generate_input.py -output input1_tmp.fits -prefix ../ARs_dataset/SDO_vz1 -nb_frames 31 -first_frame 0
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python deepvel_generate_input.py -output input2_tmp.fits -prefix ../ARs_dataset/SDO_bz1 -nb_frames 31 -first_frame 0
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python deepvel.py -i1 input1_tmp.fits -i2 input2_tmp.fits -o output_MURaM_test_set/deepvel_output_series_00-31.fits -bx1 0 -bx2 11 by1 0 -by2 11 -sim 1 -n network &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
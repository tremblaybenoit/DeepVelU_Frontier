python deepvel_generate_input.py -output input_tmp.fits -prefix path_to_data_plus_prefix -nb_frames 31 -first_frame 0
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
python deepvel.py -i input_tmp.fits -o output_SDO_test_set/deepvel_output_series_00-31.fits -bx1 0 -bx2 11 -by1 0 -by2 11 -median 0 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
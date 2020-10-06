python test.py --method=StainNet --test_ssim --model_path=checkpoints/stainnet/Histopathology/stainnet_staingan_hist_best_psnr_layer3_ch32.pth  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=StainGAN --test_ssim --model_path=checkpoints/staingan/Histopathology/latest_net_G_A.pth  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=reinhard --test_ssim --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=reinhard --test_ssim --random_target  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=macenko  --test_ssim  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=macenko  --test_ssim --random_target  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=vahadane --test_ssim  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=vahadane --test_ssim --random_target  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=khan --test_ssim --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB
python test.py --method=khan --test_ssim --random_target  --source_dir=dataset/Histopathology/test/testA --gt_dir=dataset/Histopathology/test/testB


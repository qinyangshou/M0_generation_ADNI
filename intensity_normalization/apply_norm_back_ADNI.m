addpath(genpath('toolbox'))
scale_path = '../input_data';
model_pred_base = '../generated_data'; 
model_name = '2025-02-14T12-08-21_finetune_Mayo_combined';
model_path = [model_pred_base, model_name];

subjects = dir(fullfile(model_path,'*_S_*'));
subjects = subjects(1:end);

for i = 1:length(subjects)
    sub = subjects(i).name;
    disp(sub)
    tmp = load_untouch_nii(fullfile(model_path, sub, 'generated_M0_avg20.nii'));
    gen_M0 = double(tmp.img);
    load(fullfile(scale_path, sub, 'PASL_processed','norm_scale_GW_combined.mat'))
    normback_M0 = gen_M0 * norm_scale;
    tmp.img = normback_M0;
    tmp.hdr.dime.datatype = 16;
    save_untouch_nii(tmp,fullfile(model_path, sub, 'normbackGW_generated_M0_avg20.nii'))
end
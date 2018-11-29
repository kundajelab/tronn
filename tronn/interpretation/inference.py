"""description: wrappers for inference runs
"""




# really an integrative thing between the model manager and dataloader - new module?
# from tronn.interpretation.inference import run_inference
# model manager is either a list or individual - maybe just always have it be a list,
# so wrap it when going in
def run_inference(args):
    """run inference on one model
    """
    # set up list of model managers
    if args.model == "kfolds":
        model_managers = []
        model_jsons = args.model["params"]["models"]
        for model_json in model_jsons:
            with open(model_json, "r") as fp:
                args.model = json.load(fp)
            model_manager = setup_model_manager(args)
        model_managers.append(model_manager)
    else:
        model_managers = [setup_model_manager(args)]

    # go through each one
    for model_manager in model_managers:
        pass
                

    
    
    return



def run_inference(
        data_loader,
        model_managers, # <- just always make this a list?
        inference_params,
        out_file,
        sample_size=1000,
        debug=False,
        positives_only=True,
        kfold=False):
    """wrapper for inference
    """
    for run_idx in xrange(len(model_managers)):
    
        # for loop starts here, check model manager
        
        # adjust data files if kfold, also for positives
        
        # set up input fn
        
        # also check if processed inputs
        
        # set up model(s)
        
        # add model fn to inference params
        
        # 
    
    
    return

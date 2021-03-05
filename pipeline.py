
# ----------------Pipeline overview----------------
# Load feature vectors
# Load models
# Generate cross validation folds
# Run K-fold cross validation
    # Save checkpoint
# Save results
# ----------------Pipeline overview----------------





# Log start of experiment
exp_start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Experiment "+ experiment_name +" started at ", exp_start_time, "\n")	

# save experiment
save_pickle(results, './results/'+experiment_name+'_results_bootstrap.sav')    

# Log end of experiment
print("Total time elapsed : " + str(math.floor(total_time_elapsed/60)) + "m " + str(round(total_time_elapsed % 60)) + "s")
print("Experiment started at ", exp_start_time)	
print("Experiment ended at ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))	




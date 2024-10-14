# -*- coding: utf-8 -*-
	

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
import time
from neural_net import net
from sklearn.feature_selection import chi2

warnings.filterwarnings('ignore')
		
		

####==============================================####
####============== Data Exploration ==============####
####==============================================####

def print_records(dataset):
	print("============================================")
	for row_num, row in enumerate(dataset.values):
		for col_num, column in enumerate(dataset.columns):
			print("{}: {}".format(column, row[col_num]))
		print("")
	print("============================================")
		
def convert_frame(frame):
	frame = frame.astype(float)
	return torch.tensor(frame.values, requires_grad=True)

all_models = {}
#This is a modular function for iterative exploration of hyperparameters. 
#It outputs all the results to two separate dictionaries for graphing model performance and cherry picking best models.
def optimize(training_data, validation_data, num_samples, training_count):
	cols = len(training_data.columns)-1
	validation_subset = validation_data#.sample(2048)
	validation_targets = convert_frame(validation_subset["TARGET"]).long()
	validation_subset = convert_frame(validation_subset.drop(["TARGET"], axis=1)).float()
	
	params = {}
	params["rate"] = list(10**np.random.uniform(-3, -1, 3))
	params["reg"] = list(10**np.random.uniform(-5, -3, 3))
	params["batch_size"] = [2048]	
	params["epochs"] = [10]
	params["num_layers"] = [4]
	params["layer_size"] = [50, 100, 150, 250]
	params["lr_decay"] = [0.92]
	params["pruning"] = [0.8]
	params["target_weight"] = [11.5]
	
	combinations = itertools.product(*params.values())
	total_counts = len(list(combinations))
	combinations = itertools.product(*params.values())
	print(total_counts)
	for model_num, param_set in enumerate(list(combinations)):
		start = time.time()
		param_dict = {}
		for index, param in enumerate(param_set):
			param_dict[list(params.keys())[index]] = param
		
		model = net(num_features = cols, 
		   params = param_dict,
		   verbose=False)
		
		for sample_num in range(num_samples):
			training_subset = training_data.sample(training_count)
			training_targets = training_subset["TARGET"]
			training_subset = training_subset.drop(["TARGET"], axis=1)
			model.train(training_subset, training_targets, validation_subset, validation_targets)
			
		print("[{}/{}] Time Elapsed: {:d}s Training Loss: {:.4f} Validation Loss: {:.4f}".format(model_num+1, total_counts, int(time.time()-start), model.loss_history[-1], model.validation_loss[-1]))
		
		equal = model.target_accuracy > 0.6 and model.nontarget_accuracy > 0.6
		reliable = model.nontarget_accuracy > 0.85 and model.target_accuracy > 0.3
		if True or (equal or reliable) and model.performance > 1.2:
			run_models({(param_dict["rate"], param_dict["reg"]): model}, validation_data, verbose = False)
			equal = model.target_accuracy > 0.6 and model.nontarget_accuracy > 0.6
			reliable = model.nontarget_accuracy > 0.8 and model.target_accuracy > 0.2
			if True or (equal or reliable) and model.performance > 1.2:
				print(("    ").join(["{}: {:.9}".format(key, str(value if type(value) != np.float64 else f'{value:.2e}')) for key, value in param_dict.items()]))
				print("Correct Targets: {:.2f}%".format(100*model.target_accuracy))
				print("Correct Nontargets: {:.2f}%".format(100*model.nontarget_accuracy))
				print("=======================================")
		all_models[param_set] = model

#This function accepts the raw data provided for the project and converts it into a format that can be consumed.
#It aggregates data from all tables and joins them with the main data set.
#A section of data is segmented off for use as model validation for determining generalizability.
def fix_data(training_data, testing_data, validation_size = 0.1):
	training_data = training_data.copy()
	testing_data = testing_data.copy()
	tables = FieldCalcs.get_calculated_fields()
	for name, table in tables.items():
		table.columns = [f"{name}_{column}" for column in table.columns]
		
	datasets = {"training_data": training_data, "testing_data": testing_data}
	for dataset, data in datasets.items():
		dummies = []
		for name, table in tables.items():
			data = pd.merge(data, table, on="SK_ID_CURR", how="left")
			for column in table.columns:
				if "distribution" in column:
					continue
				if table[column].dtype != object:
					data[f"flag_{column}"] = np.where((~data[column].isnull() & data[column] != 0), 1, 0)
		data["total_annuities"] = pd.merge(data[["SK_ID_CURR", "AMT_ANNUITY"]], tables["total_annuities"].fillna(0), on="SK_ID_CURR", how ="left").iloc[:, 1:].sum(axis=1)
		data["debt_to_income"] = data["total_annuities"] / data["AMT_INCOME_TOTAL"]
		
		data = data.drop(["SK_ID_CURR"], axis=1)
		for column in data.columns:
			if column == "TARGET" or column == "SK_ID_CURR":
				continue
			if data[column].dtype == object or (data[column].dtype == np.int64 and column[0:3] != "CNT" and column[0:4] != "DAYS"):
				dummies.append(column)
		
		for column in data.columns:
			if data[column].isnull().any():
				data["nan_"+column] = np.where(data[column].isnull(), 1, 0)
				if column not in dummies:
					data[column] = data[column].fillna(0)
		
		datasets[dataset] = pd.get_dummies(data, columns=dummies)
		
		
	
	cols = datasets["training_data"].columns.union(datasets["testing_data"].columns)
	training_data = datasets["training_data"].reindex(cols, axis=1, fill_value=0).astype(np.float64)
	testing_data = datasets["testing_data"].reindex(cols, axis=1, fill_value=0).astype(np.float64)
	testing_data = testing_data.drop(columns=["TARGET"], axis=1)
	
	cutoff = int(len(training_data)* (1-validation_size))
	validation_data = training_data[cutoff:]
	training_data = training_data[:cutoff]
	
	return training_data, validation_data, testing_data, tables
	


#=============================================================
#==================== Bureau Calculations ====================
#=============================================================

#All data engineering occurs here. Tables are queried for merging.
class FieldCalcs:
	
	def __init__(self):
		pass
	
	@staticmethod
	def get_calculated_fields():
		tables = {}
		aggregated_bureau = pd.DataFrame(np.unique(data_bureau["SK_ID_CURR"]), index = np.unique(data_bureau["SK_ID_CURR"]), columns = ["SK_ID_CURR"])
		
		
		tables["most_common_credit_type"] = data_bureau[["SK_ID_CURR", "CREDIT_TYPE"]].groupby(["SK_ID_CURR"], as_index = True).agg(pd.Series.mode).apply(lambda x: x[0] if type(x[0]) != np.ndarray else x[0][0], axis=1)
		tables["most_common_credit_type"] = pd.DataFrame(tables["most_common_credit_type"], columns=["credit_type"])
		tables["bureau_loans"] = data_bureau.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"], as_index=True)[["CREDIT_ACTIVE"]].count().unstack(level=1).fillna(0)
		tables["bureau_loan_totals"] = data_bureau.groupby(["SK_ID_CURR"], as_index=True)[["SK_ID_CURR"]].count()
		tables["bureau_status_distribution"] = tables["bureau_loans"]/tables["bureau_loan_totals"].values
		
		#============================================================
		#================= Home Credit Calculations =================
		#============================================================
		
		
		tables["credit_loans"] = data_previous.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"], as_index=True)[["SK_ID_CURR"]].count().unstack(level=1)
		tables["credit_loans"].columns = tables["credit_loans"].columns.droplevel(0)
		tables["total_loans"] = data_previous.groupby(["SK_ID_CURR"], as_index=True)[["SK_ID_CURR"]].count()
		tables["credit_loan_distribution"] = tables["credit_loans"] / tables["total_loans"].values
		tables["uses_loan_insurance"] = data_previous.groupby(["SK_ID_CURR"], as_index=True)[["NFLAG_INSURED_ON_APPROVAL"]].sum().rename(columns = {"NFLAG_INSURED_ON_APPROVAL": "uses_insurance"})
		tables["uses_loan_insurance"]["uses_insurance"] = np.where(tables["uses_loan_insurance"]["uses_insurance"] > 0, 1, 0)
		
		rejected_priors = data_previous[(data_previous["CODE_REJECT_REASON"] != "XAP") & (data_previous["CODE_REJECT_REASON"] != "XNA")]
		
		#merge the following
		tables["rejected_loans"] = rejected_priors.groupby(["SK_ID_CURR", "CODE_REJECT_REASON"], as_index=True)[["SK_ID_CURR"]].count().unstack(level=1).fillna(0)
		tables["rejected_loans"].columns = tables["rejected_loans"].columns.droplevel(0)
		tables["total_rejected_loans"] = rejected_priors.groupby(["SK_ID_CURR"], as_index=True)[["SK_ID_CURR"]].count()
		tables["rejected_loan_distribution"] = tables["rejected_loans"] / tables["total_rejected_loans"].values
		
		tables["client_type"] = data_previous.groupby("SK_ID_CURR")[["DAYS_DECISION"]].max()
		tables["client_type"] = tables["client_type"].merge(data_previous, on=["SK_ID_CURR", "DAYS_DECISION"], how="left").drop_duplicates(subset=["SK_ID_CURR", "DAYS_DECISION"])
		tables["client_type"] = tables["client_type"][["SK_ID_CURR", "NAME_CLIENT_TYPE"]].set_index("SK_ID_CURR")
		
		for column in data_previous.dtypes[data_previous.dtypes == object].index:
			if column in ["FLAG_LAST_APPL_PER_CONTRACT", "CODE_REJECT_REASON", "NAME_CLIENT_TYPE"]:
				continue
			col_lower = column.lower()
			tables[col_lower] = data_previous.groupby(["SK_ID_CURR", column])[[column]].count().unstack(level=1)
			tables[col_lower].columns = tables[col_lower].columns.droplevel(0)
			tables[f"distribution_{col_lower}"] = tables[col_lower] / data_previous.groupby("SK_ID_CURR")[["SK_ID_CURR"]].count()
		#============================================================
	 	#=================== Debt to Income Ratio ===================
	 	#============================================================
		
		bureau_debts = data_bureau[data_bureau["CREDIT_ACTIVE"] == "Active"]
		bureau_debts.index = bureau_debts.SK_ID_BUREAU
		bureau_debts = bureau_debts[(bureau_debts["AMT_CREDIT_SUM_DEBT"] > 0)]
		
		tables["bureau_debt_totals"] = bureau_debts.groupby(["SK_ID_CURR"])[["AMT_CREDIT_SUM_DEBT"]].sum()
		
		perpetuities = bureau_debts[(bureau_debts["AMT_CREDIT_SUM_DEBT"] > 0) & (bureau_debts["AMT_ANNUITY"] == 0)].set_index("SK_ID_BUREAU")
		perpetuity_annuities = (perpetuities["AMT_CREDIT_SUM"] - perpetuities["AMT_CREDIT_SUM_DEBT"])/(perpetuities["DAYS_CREDIT"] / -30)
		perpetuity_annuities = np.clip(perpetuity_annuities, a_min = 0, a_max = None)
		bureau_debts.loc[perpetuities.index, "AMT_ANNUITY"] = perpetuity_annuities
		
		bureau_debts = bureau_debts.groupby("SK_ID_CURR")[["AMT_ANNUITY"]].sum()
		
		#payments = data_cash.groupby(["SK_ID_CURR", "SK_ID_PREV"])[["MONTHS_BALANCE"]].max().reset_index()
		#prior_loans = pd.merge(payments, data_cash[["SK_ID_PREV", "NAME_CONTRACT_STATUS", "MONTHS_BALANCE"]], on=["SK_ID_PREV", "MONTHS_BALANCE"], how="left")
		#active_priors = prior_loans[prior_loans.NAME_CONTRACT_STATUS == "Active"]
		active_debts = data_previous[data_previous.DAYS_TERMINATION > 0]
		active_debts = active_debts.groupby(["SK_ID_CURR"])[["AMT_ANNUITY"]].sum()
		
		tables["total_annuities"] = pd.merge(bureau_debts, active_debts, on="SK_ID_CURR", how="outer")
		
		
		#categorical_columns = [col_name for col_name in col_names if col_name[0:8] == "category"]
		#results = pd.get_dummies(results, columns = categorical_columns)
		#results = results.drop(categorical_columns)
		return tables

#==================================================

#Out of the cherry picked models from the optimize() function, they will be evaluated for best performance on validation data.
def run_models(models, val_data, verbose = True, split_cnt = 1):
	val_targets = val_data["TARGET"]
	val_data = val_data.drop(["TARGET"], axis=1)
	
	val_targets = convert_frame(val_targets).long()
	val_data = convert_frame(val_data).float()
	
	
	best_model = None
	
	data_splits = torch.tensor_split(val_data, split_cnt)
	target_splits = torch.tensor_split(val_targets, split_cnt)

	
	for settings, model in models.items():
		target_accuracy = []
		nontarget_accuracy = []
		performance = []
		split_sets = zip(data_splits, target_splits)
		for val_data, val_targets in split_sets:
			nontargets = sum(val_targets==0)
			targets = sum(val_targets == 1)
			scores = model.get_accuracy(val_data, val_targets)
			target_scores = sum(x > 0.5 and y == 1 for x, y in zip(scores[:, 0], val_targets))
			nontarget_scores = sum(x < 0.5 and y == 0 for x, y in zip(scores[:, 0], val_targets))
		
			target_accuracy.append(target_scores/targets)
			nontarget_accuracy.append(nontarget_scores/nontargets)
			performance.append((target_scores/targets)+(nontarget_scores/nontargets))
			
		model.target_accuracy = np.mean(target_accuracy)
		model.nontarget_accuracy = np.mean(nontarget_accuracy)
		model.performance = np.mean(performance)
		if verbose: 
			print("Target percentage: {:.2f}%".format(100*model.target_accuracy))
			print("Nontarget percentage: {:.2f}%".format(100*model.nontarget_accuracy))
		if best_model == None or best_model.performance < model.performance:
			best_model = model
			best_model.settings = settings
		
	if verbose: 
		print("Best Results:")
		print("Target Accuracy: {:.2f}%".format(100*best_model.target_accuracy))
		print("Nontarget Accuracy: {:.2f}%".format(100*best_model.nontarget_accuracy))
	
	return best_model

#When a model is ready to run predictive analysis for final evaluation, this function will yield a csv containing applicant classifications.
def print_submission(model, test_ids, test_data):
	test_data = convert_frame(test_data).float()
	scores = model.get_accuracy(test_data, targets = None, test=True)
	print(scores)
	scores = scores[:, 0].tolist()
	#scores = np.argmax(scores, axis=1)
	file = open(r"predictions.csv", "w")
	file.write("SK_ID_CURR,TARGET\n")
	for test_id, score in zip(test_ids, scores):
		file.write("{},{}\n".format(test_id, score))
		
	file.close()


#Raw data imports
data_bureau = pd.read_csv(r"bureau.csv")
data_balance = pd.read_csv(r"bureau_balance.csv")
data_credit = pd.read_csv(r"credit_card_balance.csv")
data_payments = pd.read_csv(r"installments_payments.csv")
data_cash = pd.read_csv(r"POS_CASH_balance.csv")
data_previous = pd.read_csv(r"previous_application.csv")
data_train = pd.read_csv(r"application_train.csv")
data_test = pd.read_csv(r"application_test.csv")



#Final tables of parsed data inputs
training_data, validation_data, testing_data, tables = fix_data(data_train, data_test, validation_size = 0.1)



chi2_results = chi2(training_data.drop("TARGET", axis =1).apply(lambda x: np.abs(x)).values, training_data.TARGET)
drop_columns = training_data.drop("TARGET", axis = 1).columns[chi2_results[1] > 0.0001]
training_selection = training_data.drop(drop_columns, axis=1)
validation_selection = validation_data.drop(drop_columns, axis=1)
testing_selection = testing_data.drop(drop_columns, axis=1)


optimize(training_selection, validation_selection, 4, 20480)
best_model = run_models(all_models, validation_selection, split_cnt = 3)
print_submission(best_model, data_test.SK_ID_CURR, testing_selection)

#Tool for visually evaluating resulting models performance in training via graph
def print_loss_graph(results, title):
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.set_ylim([0, 2])
	for result in results:
		model = results[result]
		ax.plot(torch.tensor(model.loss_history).tolist(), color="tab:blue")
		ax.plot(torch.tensor(model.validation_loss).tolist(), color="tab:orange")
	fig
	
fig, ax = plt.subplots()
ax.set_ylim([1,1.5])
for params, model in all_models.items():
    ax.plot([model.model_params["layer_size"]], [model.performance], "bo")
ax.plot([200,1200], [1.23, 1.23], linestyle="-", linewidth=2)
fig

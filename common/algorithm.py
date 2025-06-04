from datasets.utils import build_dataloader
from trainers.get_trainer import get_trainer
from common.utils import save_exp_result


def cp(args):
    for i in range(args.num_runs):
        train_loader, cal_loader, _, test_loader = build_dataloader(args)

        trainer = get_trainer(args)

        trainer.train(train_loader, args.epochs)

        trainer.predictor.calibrate(cal_loader)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)

def tune(args):
    holdout_covgap = []
    tune_covgap = []
    tuning_bias_list = []
    holdout_coverage = []
    tune_coverage = []

    for i in range(args.num_runs):
        train_loader, cal_loader, tune_loader, test_loader = build_dataloader(args)

        trainer = get_trainer(args)

        trainer.train(train_loader, args.epochs)

        trainer.model.tune(tune_loader)

        trainer.predictor.calibrate(cal_loader)
        holdout_result_dict = trainer.predictor.evaluate(test_loader)

        trainer.predictor.calibrate(tune_loader)
        tune_result_dict = trainer.predictor.evaluate(test_loader)

        tuning_bias = abs(tune_result_dict["Coverage"] - (1 - args.alpha)) - abs(
            holdout_result_dict["Coverage"] - (1 - args.alpha))
        final_result_dict = {"TuningBias": tuning_bias}

        print("Using holdout calibration set: ")
        for key, value in holdout_result_dict.items():
            print(f'{key}: {value}')
        print("Using same set for calibration and tune: ")
        for key, value in tune_result_dict.items():
            print(f'{key}: {value}')
        print("Tuning Bias: ", tuning_bias)
        print()
        for key, value in holdout_result_dict.items():
            final_result_dict["holdout_" + key] = value

        for key, value in tune_result_dict.items():
            final_result_dict["tune_" + key] = value

        holdout_covgap.append(abs(holdout_result_dict["Coverage"] - (1 - args.alpha)))
        tune_covgap.append(abs(tune_result_dict["Coverage"] - (1 - args.alpha)))
        holdout_coverage.append(holdout_result_dict["Coverage"])
        tune_coverage.append(tune_result_dict["Coverage"])
        tuning_bias_list.append(tuning_bias)

def standard(args):
    for i in range(args.num_runs):
        train_loader, _, _, test_loader = build_dataloader(args)


        trainer = get_trainer(args)

        trainer.train(train_loader, args.epochs)


        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')

        if args.save == "True":
            save_exp_result(args, result_dict)
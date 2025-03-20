import scores


def get_score(args):
    if args.score == "thr":
        return scores.thr.THR()
    elif args.random is None:
        raise ValueError("Please specify --random.")
    elif args.score == "aps":
        return scores.aps.APS((args.random == "True"))
    elif args.score == "raps":
        if args.raps_size_penalty_weight is None or args.size_regularization is None:
            raise ValueError("Please specify --raps_size_penalty_weight and --size_regularization.")

        return scores.raps.RAPS((args.random == "True"),
                                               weight=args.raps_size_penalty_weight,
                                               size_regularization=args.size_regularization)
    elif args.score == "saps":
        if args.saps_size_penalty_weight is None:
            raise ValueError("Please specify --saps_size_penalty_weight.")

        return scores.saps.SAPS((args.random == "True"),
                                               weight=args.saps_size_penalty_weight, )
    return RuntimeError("Can not find a suitable score function.")

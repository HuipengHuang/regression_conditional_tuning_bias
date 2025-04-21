import scores


def get_score(args):
    if args.score == "thr":
        return scores.thr.THR()

    elif args.random is None:
        # APS, RAPS, SAPS all need to choose random version or determined version.
        assert args.random is not None, "Please specify --random."

    elif args.score == "aps":
        return scores.aps.APS((args.random == "True"))

    elif args.score == "raps":
        assert args.raps_size_penalty_weight is not None, "Please specify --raps_size_penalty_weight."
        assert args.raps_size_regularization is not None, "Please specify --size_regularization."
        assert args.raps_size_penalty_weight >= 0, "raps_size_penalty_weight must be greater than or equal to 0."
        assert args.raps_size_regularization >= 0, "raps_size_regularization must be greater than or equal to 0."

        return scores.raps.RAPS(
            (args.random == "True"),
            weight=args.raps_size_penalty_weight,
            size_regularization=args.raps_size_regularization
        )

    elif args.score == "saps":
        # Validate required arguments for SAPS
        assert args.saps_size_penalty_weight is not None, "Please specify --saps_size_penalty_weight."
        assert args.saps_size_penalty_weight >= 0, "saps_size_penalty_weight must be greater than or equal to 0."

        return scores.saps.SAPS(
            (args.random == "True"),
            weight=args.saps_size_penalty_weight,
        )
    elif args.score == "weight_score":
        return scores.weighted_score.WeightedScore()

    # If no valid score function is found
    raise RuntimeError("Cannot find a suitable score function.")

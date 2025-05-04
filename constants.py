"""
Constants and configurations for options trading strategies.
"""

# Focus strategies - these are the strategies the system will prioritize
FOCUS_STRATEGIES = [
    "IRON CONDOR",
    "BULL PUT",
    "CALL VERTICAL",
    "PUT VERTICAL",
    "CALL BUTTERFLY",
    "PUT BUTTERFLY",
    "IRON BUTTERFLY",
    "CALL CALENDAR",
    "PUT CALENDAR",
    "DOUBLE CALENDAR",
    "CALL DIAGONAL",
    "PUT DIAGONAL",
    "DOUBLE DIAGONAL",
    "STRADDLE",
    "CALL BACKRATIO",
    "PUT BACKRATIO",
    "CALL",
    "PUT",
    "SHORT PUT"
]

# Number of unique strikes required per strategy
STRATEGY_UNIQUE_STRIKES = {
    "IRON CONDOR": 4,
    "BULL PUT": 2,
    "CALL VERTICAL": 2,
    "PUT VERTICAL": 2,
    "CALL BUTTERFLY": 3,
    "PUT BUTTERFLY": 3,
    "IRON BUTTERFLY": 3,
    "CALL CALENDAR": 1,
    "PUT CALENDAR": 1,
    "DOUBLE CALENDAR": 2,
    "CALL DIAGONAL": 2,
    "PUT DIAGONAL": 2,
    "DOUBLE DIAGONAL": 4,
    "STRADDLE": 1,
    "CALL BACKRATIO": 2,
    "PUT BACKRATIO": 2,
    "CALL": 1,
    "PUT": 1,
    "SHORT PUT": 1
}

# Leg definitions for each strategy
STRATEGY_LEGS = {
    "IRON CONDOR": [
        {"type": "Put", "position": "Long"},
        {"type": "Put", "position": "Short"},
        {"type": "Call", "position": "Short"},
        {"type": "Call", "position": "Long"}
    ],
    "BULL PUT": [
        {"type": "Put", "position": "Short"},
        {"type": "Put", "position": "Long"}
    ],
    "CALL VERTICAL": [
        {"type": "Call", "position": "Long"},
        {"type": "Call", "position": "Short"}
    ],
    "PUT VERTICAL": [
        {"type": "Put", "position": "Short"},
        {"type": "Put", "position": "Long"}
    ],
    "CALL BUTTERFLY": [
        {"type": "Call", "position": "Long"},
        {"type": "Call", "position": "Short (2 contracts)"},
        {"type": "Call", "position": "Long"}
    ],
    "PUT BUTTERFLY": [
        {"type": "Put", "position": "Long"},
        {"type": "Put", "position": "Short (2 contracts)"},
        {"type": "Put", "position": "Long"}
    ],
    "IRON BUTTERFLY": [
        {"type": "Put", "position": "Long"},
        {"type": "Put", "position": "Short"},
        {"type": "Call", "position": "Short"},
        {"type": "Call", "position": "Long"}
    ],
    "CALL CALENDAR": [
        {"type": "Call", "position": "Short (near-term expiration)"},
        {"type": "Call", "position": "Long (far-term expiration)"}
    ],
    "PUT CALENDAR": [
        {"type": "Put", "position": "Short (near-term expiration)"},
        {"type": "Put", "position": "Long (far-term expiration)"}
    ],
    "DOUBLE CALENDAR": [
        {"type": "Put", "position": "Short (near-term expiration)"},
        {"type": "Put", "position": "Long (far-term expiration)"},
        {"type": "Call", "position": "Long (far-term expiration)"},
        {"type": "Call", "position": "Short (near-term expiration)"}
    ],
    "CALL DIAGONAL": [
        {"type": "Call", "position": "Short (near-term expiration)"},
        {"type": "Call", "position": "Long (far-term expiration)"}
    ],
    "PUT DIAGONAL": [
        {"type": "Put", "position": "Short (near-term expiration)"},
        {"type": "Put", "position": "Long (far-term expiration)"}
    ],
    "DOUBLE DIAGONAL": [
        {"type": "Put", "position": "Short (near-term expiration)"},
        {"type": "Put", "position": "Long (far-term expiration)"},
        {"type": "Call", "position": "Long (far-term expiration)"},
        {"type": "Call", "position": "Short (near-term expiration)"}
    ],
    "STRADDLE": [
        {"type": "Call", "position": "Long"},
        {"type": "Put", "position": "Long"}
    ],
    "CALL BACKRATIO": [
        {"type": "Call", "position": "Short"},
        {"type": "Call", "position": "Long (2 contracts)"}
    ],
    "PUT BACKRATIO": [
        {"type": "Put", "position": "Short"},
        {"type": "Put", "position": "Long (2 contracts)"}
    ],
    "CALL": [
        {"type": "Call", "position": "Long"}
    ],
    "PUT": [
        {"type": "Put", "position": "Long"}
    ],
    "SHORT PUT": [
        {"type": "Put", "position": "Short"}
    ]
}

# Risk-reward constraints per strategy
STRATEGY_RISK_CONSTRAINTS = {
    "IRON CONDOR": {"max_risk_reward_ratio": 2.5, "min_pop": 0.65},
    "BULL PUT": {"max_risk_reward_ratio": 2.0, "min_pop": 0.6},
    "CALL VERTICAL": {"max_risk_reward_ratio": 2.0, "min_pop": 0.55},
    "PUT VERTICAL": {"max_risk_reward_ratio": 2.0, "min_pop": 0.55},
    "CALL BUTTERFLY": {"max_risk_reward_ratio": 5.0, "min_pop": 0.25},
    "PUT BUTTERFLY": {"max_risk_reward_ratio": 5.0, "min_pop": 0.25},
    "IRON BUTTERFLY": {"max_risk_reward_ratio": 4.0, "min_pop": 0.3},
    "CALL CALENDAR": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "PUT CALENDAR": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "DOUBLE CALENDAR": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "CALL DIAGONAL": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "PUT DIAGONAL": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "DOUBLE DIAGONAL": {"max_risk_reward_ratio": 1.5, "min_pop": 0.5},
    "STRADDLE": {"max_risk_reward_ratio": 3.0, "min_pop": 0.4},
    "CALL BACKRATIO": {"max_risk_reward_ratio": 1.0, "min_pop": 0.45},
    "PUT BACKRATIO": {"max_risk_reward_ratio": 1.0, "min_pop": 0.45},
    "CALL": {"max_risk_reward_ratio": 5.0, "min_pop": 0.35},
    "PUT": {"max_risk_reward_ratio": 5.0, "min_pop": 0.35},
    "SHORT PUT": {"max_risk_reward_ratio": 5.0, "min_pop": 0.65}
}

# Market condition indicators for strategy selection
MARKET_CONDITIONS = {
    "HIGH_VOLATILITY": {
        "threshold": 0.25,  # IV > 25%
        "favorable_strategies": ["IRON CONDOR", "IRON BUTTERFLY", "SHORT PUT"]
    },
    "LOW_VOLATILITY": {
        "threshold": 0.15,  # IV < 15%
        "favorable_strategies": ["CALL CALENDAR", "PUT CALENDAR", "DOUBLE CALENDAR"]
    },
    "BULLISH": {
        "favorable_strategies": ["CALL VERTICAL", "BULL PUT", "CALL BACKRATIO"]
    },
    "BEARISH": {
        "favorable_strategies": ["PUT VERTICAL", "PUT BACKRATIO", "CALL CALENDAR"]
    },
    "NEUTRAL": {
        "favorable_strategies": ["IRON CONDOR", "IRON BUTTERFLY", "DOUBLE CALENDAR"]
    },
    "VOLATILE": {
        "favorable_strategies": ["STRADDLE", "CALL", "PUT"]
    }
}

# Default training parameters
TRAINING_PARAMS = {
    "gamma": 0.99,
    "lr": 3e-4,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01
}

# Default evaluation parameters
EVAL_PARAMS = {
    "min_trades": 10,
    "eval_episodes": 5
}

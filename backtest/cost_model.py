from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    fee: float = 0.0005  # per side, research assumption rather than account fee tier
    slippage: float = 0.0003  # per side

    def __post_init__(self):
        if not 0 <= self.fee < 1 or not 0 <= self.slippage < 1:
            raise ValueError('Costs must be fractions in [0,1)')

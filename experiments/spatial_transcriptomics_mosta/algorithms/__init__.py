from . import ours, sgd, adamw, rmsprop, cv, svi_constant, svi_faster

ALGORITHMS = {
    "ours": ours.run,
    "sgd": sgd.run,
    "adamw": adamw.run,
    "rmsprop": rmsprop.run,
    "cv": cv.run,
    "svi_constant": svi_constant.run,
    "svi_faster": svi_faster.run,
}

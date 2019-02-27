from eggtart import Eggtart, eggtart_optimizer
# from challah import Challah, challah_optimizer
# from sourdough import Sourdough, sourdough_optimizer
# from bublik import Bublik, bublik_optimizer
from rolls import Rolls, rolls_optimizer

# discriminators
disc = {
    'eggtart': Eggtart,
    # 'sourdough': Sourdough,
    # 'challah': Challah,
    # 'bublik': Bublik,
}

# Optimizers
optimsDisc = {
    'eggtart': eggtart_optimizer,
    # 'sourdough': sourdough_optimizer,
    # 'challah': challah_optimizer,
    # 'bublik': bublik_optimizer,
}

# generators
gen = {
    'rolls': Rolls,
}

optimsGen = {
    'rolls': rolls_optimizer,
}

from eggtart import Eggtart, eggtart_optimizer
# from challah import Challah, challah_optimizer
# from sourdough import Sourdough, sourdough_optimizer
# from bublik import Bublik, bublik_optimizer
from matzah import Matzah, matzah_optimizer

# Discriminators
disc = {
    'eggtart': Eggtart,
    # 'sourdough': Sourdough,
    # 'challah': Challah,
    # 'bublik': Bublik,
}

# Discriminator optimizers
optimsDisc = {
    'eggtart': eggtart_optimizer,
    # 'sourdough': sourdough_optimizer,
    # 'challah': challah_optimizer,
    # 'bublik': bublik_optimizer,
}

# Generators
gen = {
    'matzah': Matzah,
}

# Generator optimizers
optimsGen = {
    'matzah': matzah_optimizer,
}

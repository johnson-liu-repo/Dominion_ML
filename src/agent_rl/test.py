from pyminion_master.pyminion.bots.examples import BigMoney, BigMoneyUltimate
from pyminion_master.pyminion.expansions.base import base_set, smithy
from pyminion_master.pyminion.game import Game
from pyminion_master.pyminion.simulator import Simulator


def test_sim():
    bm = BigMoney()
    bm_ultimate = BigMoneyUltimate()
    game = Game(
        players=[bm, bm_ultimate],
        expansions=[base_set],
        kingdom_cards=[smithy]
    )
    sim = Simulator(game, iterations=2)
    result = sim.run()
    print(result)

    assert "ran 2 games" in str(result)

if __name__ == "__main__":
    test_sim()

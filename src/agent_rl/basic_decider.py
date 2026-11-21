


from typing import Protocol

from pyminion_master.pyminion.core import Card
from pyminion_master.pyminion.player import Player
from pyminion_master.pyminion.effects import Effect
from pyminion_master.pyminion.game import Game


class BasicDeciderProtocol(Protocol):

    def action_phase_decision(
            self,
            valid_actions: list["Card"],
            player: "Player",
            game: "Game",
        ) -> "Card|None":
            raise NotImplementedError("action_phase_decision is not implemented")
    
    def treasure_phase_decision(
        self,
        valid_treasures: list["Card"],
        player: "Player",
        game: "Game",
    ) -> list["Card"]:
        raise NotImplementedError("treasure_phase_decision is not implemented")

    def buy_phase_decision(
        self,
        valid_cards: list["Card"],
        player: "Player",
        game: "Game",
    ) -> "Card|None":
        raise NotImplementedError("buy_phase_decision is not implemented")





class BasicDecider():


    def action_phase_decision(
            self,
            valid_actions, # Note that valid_actions refers to RL, not in-game actions.
            player,
            game
    ):
        """
        This currently tells the bot to play no actions during the Action Phase.
        """

        return None

    def treasure_phase_decision(
            self,
            valid_actions, # Note that valid_actions refers to RL, not in-game actions.
            player,
            game
    ):
        """
        This currently tells the bot to play all available treasures in hand during the Treasure Phase.
        """
        
        return valid_actions

    def buy_phase_decision(
            self,
            valid_actions, # Note that valid_actions refers to RL, not in-game actions.
            player,
            game
    ):
        """
        This currently tells the bot to buy nothing during the Buy Phase.
        """
        
        return None



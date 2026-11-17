


import numpy as np


class ToyGameTraining:
    '''
        states:
            (L) low hand value,
            (H) high hand value,
            (T) terminal
        actions:
            (B) buy treasure,
            (S) buy score
    '''

    #######################################

    STATES  = { "L": 0, "H": 1, "T": 9 }
    ACTIONS = { "B": 0, "S": 1 }

    def __init__( self,
                  alpha: float = 0.5,
                  gamma: float = 0.9,
                  epsilon: float = 0.1 ):

        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        self.Q_table: list[list[float]] = [ [ 0.0, 0.0 ],
                                            [ 0.0, 0.0 ] ]

        self.state = self.STATES[ "L" ]

        # Action (a) in State (s) --> next state (s)
        # (B) in (L) --> (H)
        # (S) in (L) --> (L)
        # (B) in (H) --> (H)
        # (S) in (H) --> (T)

        self.transition_table = [
            [ self.STATES["H"], self.STATES["L"] ],
            [ self.STATES["H"], self.STATES["T"] ]
        ]

        self.rewards = [ [ 0, 1 ],
                         [ 0, 3 ] ]

    # ----------------------------------------------------------

    def compute_target( self,
                        reward: float,
                        next_state: int ) -> float:

        if next_state == self.STATES[ "T" ]:
            max_next_q = 0
        else:
            max_next_q = max( self.Q_table[ next_state ] )
    
        target = reward + self.gamma * max_next_q
    
        return target

    def update_Q_table( self,
                        state: int,
                        next_state: int,
                        action: int ) -> None:
        
        Q = self.Q_table[state][action]
        reward = self.rewards[state][action]
        target = self.compute_target( reward, next_state )

        new_Q = ( 1 - self.alpha ) * Q + self.alpha * target

        self.Q_table[state][action] = new_Q

    # ----------------------------------------------------------

    def take_action( self,
                     state: int,
                     action: int ) -> None:

        next_state = self.transition_table[state][action]

        self.update_Q_table( state, next_state, action )

        self.state = next_state


    # ----------------------------------------------------------

    def learn( self,
               episodes: int = 10) -> None:

        self.Q_table = [ [ 0.0, 0.0 ],
                         [ 0.0, 0.0 ] ]

        for e in range(episodes):
            self.state = self.STATES[ "L" ]
        
            while self.state != self.STATES[ "T" ]:
                ran = np.random.default_rng().random()
                # print(f"ran: {ran}")
                
                if ran <= self.epsilon:
                    action = int( np.random.default_rng().choice( list( self.ACTIONS.values() ) ) )
                else:
                    action = self.Q_table[ self.state ].index( max( self.Q_table[ self.state ] ) )

                # print(f"ep: {e}, action: {action}")

                self.take_action( self.state, action )


if __name__ == "__main__":

    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.1

    training_1 = ToyGameTraining( alpha, gamma, epsilon )

    training_1.learn(episodes=100)
    print(training_1.Q_table)

    ### It seems like the Q-values for [ [ [L, B], [L, S] ], [ [H, B], [H, S] ] ] approach
    ### [[8.999999999999979, 9.999999999999979], [9.999999999999979, 3.0]]
    ### Wait ... But doesn't this mean the agent would want to keep buying economic cards
    ### forever since the Q-value for staying in the high value state is greater than the 
    ### that for transitioning into the terminal state?
    ### Because this is too simple of a toy model / contrived example.
    ### Better results when using the rewards [ [-1, 1], [0, 3] ]

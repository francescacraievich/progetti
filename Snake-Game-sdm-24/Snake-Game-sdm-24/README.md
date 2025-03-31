# Snake-Game-sdm-24
Git repository for the final project assignment in Software Development Methods course

# Thing to do
- Method for taking user inputs
- tests
- thread for game main loop
- modify Cell to remove head and body division
- pseudocode for game loop

# Pseudocode for gameplay loop
```java
public class Pseudocode {
    public void thingToDoEachGameTick() {
        Direction newDirection = keyEvent.vector; // o comunque prendo la direzione del keyEvent
        snakeMovement.moveSnake(newDirection); // muovo il serpente
        // il controllo che il serpente abbia mangiato è gestito da moveSnake direttamente, quindi non te ne devi preoccupare qui
        // Controllo condizione di sconfitta
        if (snakeMovement.isGameOver()) {
            // Ferma tutto e mostra a schermo una finestra con scritto che hai perso :(
        }
        // Controllo condizione di vittoria, si potrebbe fare una cosa tipo
        int max_length = (snakeMovement.getBoardState().getBoardSize - 2)*(snakeMovement.getBoardState().getBoardSize - 2);
        if (snakeMovement.getSnake().getCoordSnake().size() >= max_length) {
            // Ferma tutto e mostra a schermo una finestra con scritto che hai vinto :)
        }
        // Re-render del board
    }
}
```

# Movimento
- Fare in modo che il movimento non avvenga quando si cerca di uscire dalla board
- ogni frame c'è un check delle condizioni di morte e vittoria



# Interesting concepts learnt from lecture
-  Double tests used for fake tests 
  - Dummy: never actually used (just to fill parameters)
  - Fake: have some implementation but some shortcut is taken (not suitable for production)
  - Subs: still have an implementation but not as advanced ad fake obj., returns a fixed value defined when created
  - Spies: implementation that just keeps some kinda of data/counter
  - Mocks: require a library, pre-program. obj. that focuses on the interaction between obj. and mock obj.

**things to change in the interface:** 
-the snake oppositeDirection -- it dies only when the snake size is > 2 
-incremental speed -- the game starts slow and then goes faster
-little bow on the snake with eyes
-put an unsername to remember the previous scores 

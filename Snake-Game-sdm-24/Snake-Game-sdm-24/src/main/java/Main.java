import java.awt.*;

public class Main {

    static final Coordinate BOARD_DIM = new Coordinate(11,11);

    public static void main(String[] args){
        Board board = new Board();
        SnakeMovement game = new SnakeMovement(board);

        new GameWindow(game);
    }
}

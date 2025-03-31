import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

public class ParseSnake {

    @Test
    void getCoordSnakeTest() {
        Board board = new Board("");
        board.setCell(5,6, Cell.BLANK);
        board.setCell(4,5, Cell.FOOD);
        ArrayList<Coordinate> snakeFinalCoord = new ArrayList<Coordinate>();
        snakeFinalCoord.add(0,new Coordinate(4,5));
        snakeFinalCoord.add(1,new Coordinate(5,5));
        SnakeMovement snakeMovement = new SnakeMovement(board);
        snakeMovement.setCurrentDirection(Direction.RIGHT);
        snakeMovement.moveSnake(Direction.UP);
        Snake snake = snakeMovement.getSnake();
        assertEquals(snakeFinalCoord, snakeMovement.getSnake().getCoordSnake());
    }

    @Test
    void getCoordSnakeLastPiece() {
        Board board = new Board("");
        board.setCell(5,6, Cell.BLANK);
        board.setCell(4,5, Cell.FOOD);
        Coordinate coordFinalPiece = new Coordinate(5,5);
        SnakeMovement snakeMovement = new SnakeMovement(board);
        snakeMovement.setCurrentDirection(Direction.RIGHT);
        snakeMovement.moveSnake(Direction.UP);
        Snake snake = snakeMovement.getSnake();
        assertEquals(coordFinalPiece, snakeMovement.getSnake().getCoordinateLastPieceOfBody());
    }

}

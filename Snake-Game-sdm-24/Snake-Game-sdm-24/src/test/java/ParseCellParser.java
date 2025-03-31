import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ParseCellParser {
    @Test
    void cellHasBlank() {
        Board board = new Board("");
        assertEquals(board.getCell(1,1), Cell.BLANK);
    }

    @Test
    void cellHasFood() {
        Board board = new Board("");
        assertEquals(board.getCell(5,6), Cell.FOOD);
    }

    @Test
    void cellHasSnake() {
        Board board = new Board("");
        assertEquals(board.getCell(5,5), Cell.SNAKE);
    }

    @Test
    void cellHasWall() {
        Board board = new Board("");
        assertEquals(board.getCell(0,0), Cell.WALL);
    }

    @Test
    @Disabled
    void cellHasBody() {
        Board board = new Board("");
        board.setCell(5,4, Cell.SNAKE);
        assertEquals(board.getCell(5,4), Cell.SNAKE);
    }
}

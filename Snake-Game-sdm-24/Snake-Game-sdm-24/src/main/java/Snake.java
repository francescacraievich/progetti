import java.util.ArrayList;
import java.util.Objects;

public class Snake {
    private ArrayList<Coordinate> coordSnake;

    //                                                                      //
    //  CoordBody[0] refers to the part of the body closest to the head    //
    //                                                                      //

    // length and head are deprecated, we can just use the array size for length and the position at index 0 for the head
    // We still keep the head in the Cell enum because we can use the board.getCoordinateHead()

    public Snake(ArrayList<Coordinate> coordSnake) {
        this.coordSnake = coordSnake;
    }

    public Snake(Coordinate coordHead) {
        ArrayList<Coordinate> coordSnake = new ArrayList<>();
        coordSnake.add(coordHead);
        this.coordSnake = coordSnake;
    }

    public void setCoordSnake(ArrayList<Coordinate> coordSnake) {
        this.coordSnake = coordSnake;
    }

    public Coordinate getCoordinateLastPieceOfBody() {
        int len = coordSnake.toArray().length;
        return coordSnake.get(len-1);
    }

    public ArrayList<Coordinate> getCoordSnake() {
        return this.coordSnake;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Snake snake)) return false;
        return Objects.equals(getCoordSnake(), snake.getCoordSnake());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getCoordSnake());
    }
}

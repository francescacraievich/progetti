import java.util.ArrayList;
import java.util.Objects;

public class Coordinate {
    private int x;
    private int y;

    public Coordinate(int py, int px) {
        this.x = px;
        this.y = py;
    }

    public Coordinate(int[] rowCol) {
        this.x = rowCol[1];
        this.y = rowCol[0];
    }

    public Coordinate(Coordinate coord) { //not used
        this.x = coord.getX();
        this.y = coord.getY();
    }

    // static method to get an ArrayList<Coordinate> from an int[][]
    public static ArrayList<Coordinate> createCoordinateArray(int[][] rowColArray) {
        ArrayList<Coordinate> coordArray = new ArrayList<>();
        for (int[] value : rowColArray) {
            if (value.length == 2) {
                Coordinate coord = new Coordinate(value[0], value[1]);
                coordArray.add(coord);
            } else {
                throw new StringIndexOutOfBoundsException();
            }
        }
        return coordArray;

    }

    public int getX() {
        return this.x;
    }

    public int getY() {
        return this.y;
    }

    public void setX(int x) { this.x = x; }

    public void setY(int y) { this.y = y; }

    // Changing of coordinates
    public Coordinate plus(Coordinate coordinate) {
        int tmpX = this.x + coordinate.getX();
        int tmpY = this.y + coordinate.getY();
        return new Coordinate(tmpY, tmpX);
    }

    @Override
    public String toString() {
        return "Coordinate(x=" + x + ", y=" + y + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Coordinate that)) return false;
        return getX() == that.getX() && getY() == that.getY();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getX(), getY());
    }
}

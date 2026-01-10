import org.eclipse.draw2d.*;
import org.eclipse.swt.*;

/**
 * A small example to show a red draw2d rectangle figure.
 */
public class RectangleExample {

	public static void main(String[] args) {
		Display display = new Display();
		final Shell shell = new Shell(display);
		shell.setText("Example for Draw2D");
		
		// Create LightweightSystem
		LightweightSystem lws = new LightweightSystem(shell);
	}
}

package visitor.operation.miniJava.impl;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import miniJava.visitor.miniJava.WriteImage;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.ValueOperation;
import visitor.operation.miniJava.WriteImageOperation;

public class WriteImageOperationImpl extends StatementOperationImpl implements WriteImageOperation {
	private final WriteImage it;

	private final VisitorInterface vis;

	public WriteImageOperationImpl(WriteImage it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		WriteImageOperationImpl.writeImage((WriteImage) (this.it), (String) (((ValueOperation)((ExpressionOperation)this.it.getPath().accept(vis)).evaluateExpression((State) (state)).accept(vis)).customToString()), (Value) (((ExpressionOperation)this.it.getImage().accept(vis)).evaluateExpression((State) (state))));
	}
	
	public static void writeImage(Object self, String path, Object image) {
		ArrayInstance imgInstance = ((ArrayRefValue) image).getInstance();
		int h = imgInstance.getSize();
		int w = ((ArrayRefValue) imgInstance.getValue().get(0)).getInstance().getSize();
		
		BufferedImage jImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < h; y++) {
			ArrayInstance line = ((ArrayRefValue) imgInstance.getValue().get(y)).getInstance();
			for (int x = 0; x < w; x++) {
				IntegerValue idv = (IntegerValue) line.getValue().get(x);
				int grey = idv.getValue();
				jImage.setRGB(x, y, grey+grey*256+grey*256*256);
			}
		}
		File output = new File(path);
		try {
			ImageIO.write(jImage, path.substring(path.lastIndexOf(".")+1), output);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

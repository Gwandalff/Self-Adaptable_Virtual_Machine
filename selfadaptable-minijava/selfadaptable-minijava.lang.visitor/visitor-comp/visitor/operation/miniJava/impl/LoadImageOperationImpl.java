package visitor.operation.miniJava.impl;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.LoadImage;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.LoadImageOperation;
import visitor.operation.miniJava.ValueOperation;

public class LoadImageOperationImpl extends ExpressionOperationImpl implements LoadImageOperation {
	private final LoadImage it;

	private final VisitorInterface vis;

	public LoadImageOperationImpl(LoadImage it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = ((Value) (LoadImageOperationImpl.loadImage((LoadImage) (this.it), (State) (state), (String) (((ValueOperation)((ExpressionOperation)this.it.getPath().accept(vis)).evaluateExpression((State) (state)).accept(vis)).customToString()))));
		return result;
	}
	
	public static ArrayRefValue loadImage(Object self, State state, String path) {
		BufferedImage img = null;
		try {
		    img = ImageIO.read(new File(path));
		} catch (IOException e) {
			System.out.println("ERROR loading image");
			return null;
		}
		int h = img.getHeight();
		int w = img.getWidth();
		
		MiniJavaFactory factory = MiniJavaFactory.eINSTANCE;
		
		ArrayInstance res = factory.createArrayInstance();
		res.setSize(h);
		((State) state).getArraysHeap().add(res);
		
		for (int y = 0; y < h; y++) {
			ArrayInstance line = factory.createArrayInstance();
			line.setSize(w);
			for (int x = 0; x < w; x++) {
				IntegerValue idv = factory.createIntegerValue();
				int rgb = img.getRGB(x, y);
				int b = (rgb)&0xFF;
				int g = (rgb>>8)&0xFF;
				int r = (rgb>>16)&0xFF;
				idv.setValue((r+g+b)/3);
				line.getValue().add(idv);
			}
			ArrayRefValue lineRef = factory.createArrayRefValue();
			lineRef.setInstance(line);
			res.getValue().add(lineRef);
		}
		
		ArrayRefValue ret = factory.createArrayRefValue();
		ret.setInstance(res);
		return ret;
	}
}

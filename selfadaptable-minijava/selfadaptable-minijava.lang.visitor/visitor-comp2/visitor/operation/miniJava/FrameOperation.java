package visitor.operation.miniJava;

import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Frame;

public interface FrameOperation {
	Context findCurrentContext();

	Frame findCurrentFrame();
}

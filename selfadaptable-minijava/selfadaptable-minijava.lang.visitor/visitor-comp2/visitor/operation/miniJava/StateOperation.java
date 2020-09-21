package visitor.operation.miniJava;

import java.lang.String;
import miniJava.visitor.miniJava.Call;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.ObjectInstance;

public interface StateOperation {
	Frame findCurrentFrame();

	Context findCurrentContext();

	void pushNewContext();

	void popCurrentContext();

	void println(String str);

	void pushNewFrame(ObjectInstance receiver, Call c, Context newContext);

	void popCurrentFrame();
}

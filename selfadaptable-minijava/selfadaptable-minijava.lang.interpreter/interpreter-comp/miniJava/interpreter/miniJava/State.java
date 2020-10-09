package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface State extends EObject {
	Frame getRootFrame();

	void setRootFrame(Frame value);

	EList<ObjectInstance> getObjectsHeap();

	OutputStream getOutputStream();

	void setOutputStream(OutputStream value);

	EList<ArrayInstance> getArraysHeap();

	Context getContextCache();

	void setContextCache(Context value);

	Frame getFrameCache();

	void setFrameCache(Frame value);

	Frame findCurrentFrame();

	Context findCurrentContext();

	void pushNewContext();

	void popCurrentContext();

	void println(String str);

	void pushNewFrame(ObjectInstance receiver, Call c, Context newContext);

	void popCurrentFrame();
}

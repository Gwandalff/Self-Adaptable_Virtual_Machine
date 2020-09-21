package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface State extends EObject, AcceptInterface {
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
}

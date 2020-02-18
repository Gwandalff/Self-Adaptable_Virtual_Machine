package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Frame extends EObject {
	Call getCall();

	void setCall(Call value);

	ObjectInstance getInstance();

	void setInstance(ObjectInstance value);

	Frame getChildFrame();

	void setChildFrame(Frame value);

	Frame getParentFrame();

	void setParentFrame(Frame value);

	Context getRootContext();

	void setRootContext(Context value);

	Value getReturnValue();

	void setReturnValue(Value value);

	Context findCurrentContext();

	Frame findCurrentFrame();
}
